from typing import Union, Tuple, List
import os
import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# pgps imports
from nnunetv2.training.dataloading.data_loader import DataLoader, nnUNetBaseDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
import warnings
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP

import SimpleITK as sitk

warnings.filterwarnings("ignore")

"""
############################## PGPS-curricula comment ##############################
Here we overwrite pytorch's _verify_spatial_size() function. This allows us to also use (1x1x1)-sized vectors in the bottleneck of the UNet.
Originally, pytorch would prevent normalization over 1-sized tensors. But this does not hurt/create any problems for the curriculum.
"""

import torch.nn.functional as F
F.original_verify_spatial_size = F._verify_spatial_size

def _verify_spatial_size(size: list[int]) -> None:
    # patch original function for one more patch size stage in UNet
    return

    # original method:
    # Verify that there is > 1 spatial element for instance norm calculation.
    size_prods = 1
    for i in range(2, len(size)):
        size_prods *= size[i]
    if size_prods == 1:
        raise ValueError(
            f"Expected more than 1 spatial element when training, got input size {size}"
        )

F._verify_spatial_size = _verify_spatial_size


class nnUNetDataLoader(DataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 original_batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        """
        If we get a 2D patch size, make it pseudo 3D and remember to remove the singleton dimension before
        returning the batch
        """
        super().__init__(data, batch_size, 1, None, True,
                         False, True, sampling_probabilities)

        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        # this is used by DataLoader for sampling train cases!
        self.indices = data.identifiers

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.original_batch_size = original_batch_size
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')

        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        data, seg, seg_prev, properties = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        return data_shape, seg_shape
# Start change
    def _get_rectangle_grid_shape(self, volume_shape, num_rectangles):
        max_grid_shape = [max(1, int(volume_shape[i] // self.final_patch_size[i]))
                          for i in range(len(volume_shape))]
        requested = int(num_rectangles)
        candidates = []
        for grid_shape in np.ndindex(*[limit + 1 for limit in max_grid_shape]):
            if not all(grid_shape) or np.prod(grid_shape) < requested:
                continue
            cell_count = int(np.prod(grid_shape))
            aspect = max(grid_shape) / min(grid_shape)
            candidates.append((cell_count - requested, aspect, grid_shape))

        if not candidates:
            max_rectangles = int(np.prod(max_grid_shape))
            raise ValueError(
                f'Cannot fit {requested} rectangles of patch size {self.patch_size} '
                f'into volume shape {volume_shape}; maximum is {max_rectangles}.'
            )
        return min(candidates)[2]

    def _sample_grid_bbox(self, lbs, ubs, dim, cell_id, grid_shape, grid_phase):
        """
        Sample a patch center from one rectangle without translating the grid.

        The center is sampled throughout the rectangle. Consequently, the
        patch may cross a rectangle border by up to half its patch size on each
        axis, while the global volume bounds still keep the complete patch in
        the field of view.
        """
        grid_idx = np.unravel_index(cell_id, grid_shape)
        bbox_lbs = []
        for axis in range(dim):
            volume_size = int(ubs[axis])
            num_rectangles = int(grid_shape[axis])
            model_patch_size = int(self.final_patch_size[axis])
            context_patch_size = int(self.patch_size[axis])
            rectangle_boundaries = np.floor(
                np.linspace(0, volume_size, num_rectangles + 1)
            ).astype(int)
            rectangle_lb = rectangle_boundaries[grid_idx[axis]]
            rectangle_ub = rectangle_boundaries[grid_idx[axis] + 1]
            model_patch_half = model_patch_size // 2
            first_valid_center = max(model_patch_half, rectangle_lb)
            last_valid_center = min(
                volume_size - (model_patch_size - model_patch_half), rectangle_ub - 1
            )
            if first_valid_center > last_valid_center:
                raise ValueError(
                    f'No valid patch location remains in rectangle on axis {axis}; '
                    f'volume shape {ubs}, model patch size {self.final_patch_size}.'
                )
            center = np.random.randint(first_valid_center, last_valid_center + 1)
            bbox_lbs.append(center - context_patch_size // 2)
        return bbox_lbs


    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False, old_crop: bool = False,
                cell_id: int = 0, grid_shape: Tuple[int, ...] = None, grid_phase: np.ndarray = None):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        if grid_shape is None:
            grid_shape = tuple([1] * dim)
        if grid_phase is None:
            grid_phase = np.random.uniform(0, 1, size=dim)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # Random patches are sampled from one rectangle without replacement.
        if not force_fg:
            bbox_lbs = self._sample_grid_bbox(data_shape, data_shape, dim, cell_id, grid_shape, grid_phase)
            return bbox_lbs, [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly (grid-jittered) from lb and ub. Else we need to
        # make sure we get at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = self._sample_grid_bbox(lbs, ubs, dim, cell_id, grid_shape, grid_phase)
            return bbox_lbs, [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

            bbox_lbs = self._sample_grid_bbox(lbs, ubs, dim, cell_id, grid_shape, grid_phase)
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                    'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                if not old_crop:
                    self.eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in self.eligible_classes_or_regions]
                if any(tmp):
                    if len(self.eligible_classes_or_regions) > 1:
                        self.eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(self.eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = self.eligible_classes_or_regions[np.random.choice(len(self.eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in self.eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                #jitter = [np.random.randint(-self.patch_size[i] // 4, self.patch_size[i] // 4 + 1) for i in range(dim)]
                #bbox_lbs = [max(lbs[i], min(ubs[i], (selected_voxel[i + 1] - self.patch_size[i] // 2) + jitter[i])) for i in range(dim)]

                # Convert inputs to numpy arrays for element-wise operations
                patch_size = np.array(self.patch_size)
                lbs_arr, ubs_arr = np.array(lbs), np.array(ubs)

                p_half = patch_size // 2
                base_lbs = selected_voxel[1:] - p_half
                max_j = patch_size // 4

                # Derive valid jitter bounds per axis so the patch stays within [lbs, ubs]
                min_j = np.maximum(-max_j, lbs_arr - base_lbs)
                max_j = np.minimum(max_j, ubs_arr - patch_size - base_lbs)

                # Sample valid integer jitter per axis
                jitter = np.array([
                    np.random.randint(min_j[i], max_j[i] + 1) if min_j[i] <= max_j[i] else 0 
                    for i in range(dim)
                ])

                # Final lower bounds, clamped to lbs for small volumes requiring padding
                bbox_lbs = np.maximum(lbs_arr, base_lbs + jitter).tolist()
            else:
                # If the image does not contain any foreground classes, we fall back to grid-jittered cropping
                bbox_lbs = self._sample_grid_bbox(lbs, ubs, dim, cell_id, grid_shape, grid_phase)

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs


    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        force_fg = True

        """
        ############################## PGPS-curricula comment ##############################

        here we update the batching strategy.
        default nnU-Net would create one crop/patch per patient
        this can lead to overfitting on small datasets AND also result in super expensive dataloading for very large batches
        > We create more patches/crops per patient and only load as many patients as are used for maximal patch size
        > this saves a lot of dataloading time!

        For batchsize=2 (normally used in patch-based segmentation for nnUNet) we only load 2 patients and create all crops from them
        For one patient we create only foreground patches
        For the other patient we create only background/random patches

        Background/random patches are now drawn via a grid-stratified sampler (see _sample_grid_bbox):
        the valid crop-origin range is split into a grid, a fresh random permutation of grid cells is
        drawn per loaded volume, and each crop_idx within that volume consumes one cell from the
        permutation. This guarantees every cell is used before any repeats, and a random per-volume
        phase offset keeps the grid from sitting at the same fixed alignment across volumes/epochs.
        """

        crops_per_volume = int(np.ceil(self.batch_size / self.original_batch_size))
        grid_shape = None
        grid_phase = None
        cell_order = None

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)

            if j % crops_per_volume == 0:
                if j != 0 and crops_per_volume > 1 and False:
                     # TODO: save the full volume to disk for visualization of the sampled patches and their coverage of the volume. This is useful for debugging and understanding the sampling strategy
                    os.mkdir("./debug") if not os.path.exists("./debug") else None
                    sitk.WriteImage(sitk.GetImageFromArray(crop_label_map), os.path.join("./debug", f'crop_label_map_{volume_idx}_{self.final_patch_size[0]}.nii.gz'))

                # load only every n-th volume to reduce dataloading runtime
                data, seg, seg_prev, properties = self._data.load_case(i)
                # if old_crop we can keep old bbox
                old_crop = False
                # if new patient, we want to switch to load only foreground or only background now
                force_fg = not force_fg

                # Build a volume grid with exactly one rectangle per crop.
                dim = len(data.shape[1:])
                volume_shape = data.shape[1:]
                grid_shape = self._get_rectangle_grid_shape(volume_shape, crops_per_volume)
                grid_phase = np.zeros(dim, dtype=np.int64)
                cell_order = np.random.permutation(crops_per_volume)

                if j == 0:
                    volume_idx = i
                    crop_label_map = np.zeros(data.shape, dtype=np.int16)
                    crop_label_map = crop_label_map[0]

            else:
                # here we keep on loading only foreground or background
                # if old_crop we can keep old bbox
                old_crop = True

            crop_idx_in_volume = j % crops_per_volume
            # guard in case crops_per_volume > n_cells (shouldn't happen given ceil(), but safe)
            cell_id = int(cell_order[crop_idx_in_volume % len(cell_order)])

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'], old_crop=old_crop,
                                                cell_id=cell_id, grid_shape=grid_shape, grid_phase=grid_phase)
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]
            #print(bbox)
            crop_label_map[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = 1
            
            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images


        
        
            return {'data': data_all, 'target': seg_all, 'keys': selected_keys}





        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
# End change




class nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance_GridSampler_Jitter(nnUNetTrainer):


    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size, self.original_batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size, self.original_batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val



    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network, dynamic=False) # needed for PGPS to recompile each time a new patch/batch size is started

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")



    def plan_batch_size(self):
        """
        ############################## PGPS-curricula comment ##############################
        compute maximal possible batch size for each patch size stage, such that the number of input voxels per input tensor such that GPU memory limit is never exceeded.
        Input tensor size is limited by np.prod(original batch size * original patch size)
        """

        def compute_batch_size(cur_ps, max_ps):
            cur_vox = cur_ps[0] * cur_ps[1] * cur_ps[2]
            max_vox = self.original_batch_size * max_ps[0] * max_ps[1] * max_ps[2]
            bs = int(max_vox / cur_vox)
            return bs

        cur_patch_size = self.min_patch_size
        patch_size_increment = cur_patch_size
        max_patch_size = self.original_patch_size
        patch_sizes = [cur_patch_size]
        batch_sizes = [compute_batch_size(cur_patch_size, max_patch_size)]
        patch_size_state = np.ones(3)

        # Increase the patch size such that smallest patch size dimension (numerical value) is increased first. This usually results in a higher average batchsize than increasing it naively (x>y>z>x>y>z...)
        while True:
            # first increase axis with lowest value
            i = np.argmin((cur_patch_size + patch_size_increment) * patch_size_state)
            add = np.zeros(3)
            add[i] = 1
            cur_patch_size = cur_patch_size + (add * patch_size_increment)
            if cur_patch_size[i] >= max_patch_size[i]:
                patch_size_state[i] = 1000

            bs = compute_batch_size(cur_patch_size, max_patch_size)
            batch_sizes.append(int(bs))
            patch_sizes.append(cur_patch_size.astype(int))
            print(cur_patch_size)
            if np.sum(np.where(patch_size_state == 1, 1, 0)) == 0:
                break
        self.batch_sizes = batch_sizes
        self.patch_sizes = patch_sizes

        self.print_to_log_file("All processable patchsizes:")
        self.print_to_log_file(patch_sizes)
        self.print_to_log_file("According batchsizes: ")
        self.print_to_log_file(batch_sizes)
        self.print_to_log_file("Num of patch size stages: " + str(len(batch_sizes)))


    def run_training(self):
        """
        ############################## PGPS-curricula comment ##############################
        training loop for Progressive Growing Of Patch Size:
        1. We start training on minimal possible patch size (depends on network architecture/num of pooling operations)
        2. We increase the patch size stepwise linearly during training using the minimal possible patch size steps (depends on network architecture/num of pooling operations)
        3. We use the final patch size as inference patch size

        Minimal Patch Size: minimal patch size that does result in 1x1x1 vector in bottleneck (depends on network architecture/num of pooling operations)
        Patch Size Steps: we take the minimal possible patch size update (only even integer shapes in all network stages are allowed in network processing)
        Maximal Patch Size: we take the same patch size as standard nnUNet planning (nnUNet maximizes the patch size that can fit with a batch size of 2 into a GPU memory of 11GB)
        """

        self.on_train_start()
        self.initialized = False

        self.print_to_log_file(
            '######################### Progressive Growing of Patchsize Training Config #########################')
        self.print_to_log_file('Original patch size: ' + str(self.configuration_manager.patch_size))
        self.print_to_log_file('Original batch size: ' + str(self.configuration_manager.batch_size))

        num_pool_per_axis = np.sum(np.array(self.configuration_manager.pool_op_kernel_sizes) == 2,
                                   axis=0).tolist()  # get number of pooling operations per axis

        self.print_to_log_file('num_pool_per_axis defines smallest processable patch size: ' + str(num_pool_per_axis))
        self.min_patch_size = np.array(
            [2 ** (num_pool_per_axis[0]), 2 ** (num_pool_per_axis[1]), 2 ** (num_pool_per_axis[2])])
        self.print_to_log_file(
            'Minimal possible patch size and minimal possible patch size increments: ' + str(self.min_patch_size))
        self.num_pool_per_axis = num_pool_per_axis
        self.stage = 0

        # plan patch size scheduler: which patch size increments
        self.plan_batch_size()
        self.min_patch_size = self.patch_sizes[0]

        self.print_to_log_file('Minimal USED patch size: ' + str(self.min_patch_size))

        num_stages = np.ceil(
            (self.original_patch_size[0] - self.min_patch_size[0]) / (2 ** num_pool_per_axis[0])) + np.ceil(
            (self.original_patch_size[1] - self.min_patch_size[1]) / (2 ** num_pool_per_axis[1])) + np.ceil(
            (self.original_patch_size[2] - self.min_patch_size[2]) / (2 ** num_pool_per_axis[2])) + 1
        self.print_to_log_file('Number of different patchsize phases : ' + str(num_stages))
        self.print_to_log_file(
            '######################### Progressive Growing of Patchsize Training Config #########################')

        if self.current_epoch != 0:
            # case of continuing network training
            print("continue training: from epoch: " + str(self.current_epoch))

            for completed_epoch in range(1, self.current_epoch):
                if ((completed_epoch % int(self.num_epochs / num_stages)) == 0):
                    self.stage = self.stage + 1

            print("continue with stage: " + str(self.stage))
            # increase patch size in minimal possible steps, for each patchsize-phase same number of epochs

            self.patch_size = self.patch_sizes[self.stage]
            self.batch_size = self.batch_sizes[self.stage]

            self.configuration_manager.set_patch_size(self.patch_size)
            self.print_to_log_file('New Patch Size : ' + str(self.patch_size))

            self.configuration_manager.set_batch_size(self.batch_size)
            self.print_to_log_file('New Batch Size : ' + str(self.batch_size))
            self.dataloader_train_cur_patch_size, self.dataloader_val_max_patch_size = self.get_dataloaders()

            self.initialized = True
        else:
            print("Training from scratch")

        for epoch in range(self.current_epoch, self.num_epochs):
            if not self.initialized:
                # initialize training with smallest possible patchsize (depends on num of pooling operations)

                self.patch_size = self.min_patch_size
                self.print_to_log_file('Current patchsize : ' + str(self.patch_size))
                self.configuration_manager.set_patch_size(self.patch_size)
                self.batch_size = self.batch_sizes[self.stage]
                self.configuration_manager.set_batch_size(self.batch_size)
                self.print_to_log_file('Current batchsize : ' + str(self.batch_size))
                self.dataloader_train_cur_patch_size, _ = self.get_dataloaders()
                # Maximal Patch Size Validation
                self.patch_size = self.original_patch_size
                self.configuration_manager.set_patch_size(self.patch_size)
                self.batch_size = self.original_batch_size
                self.configuration_manager.set_batch_size(self.batch_size)
                self.dataloader_train_max_patch_size, self.dataloader_val_max_patch_size = self.get_dataloaders()
                self.initialized = True

                self.patch_size = self.min_patch_size

            elif ((self.current_epoch % int(self.num_epochs / num_stages)) == 0):

                self.stage = self.stage + 1

                if self.stage == len(self.patch_sizes):
                    self.stage = self.stage - 1

                # increase patch size in minimal possible steps, for each patchsize-phase around same number of epochs (1000 epochs / number of possible patch sizes)
                # final maximal patch size is trained for residuals (1000 epochs / number of possible patch sizes + 1000 epochs % number of possible patch sizes)

                self.patch_size = self.patch_sizes[self.stage]
                current_patch_size = self.patch_size
                self.batch_size = self.batch_sizes[self.stage]

                self.configuration_manager.set_patch_size(self.patch_size)
                self.print_to_log_file("Starting a new patch size phase!")
                self.print_to_log_file('New Patch Size : ' + str(self.patch_size))

                self.configuration_manager.set_batch_size(self.batch_size)
                self.print_to_log_file('New Batch Size : ' + str(self.batch_size))
                self.dataloader_train_cur_patch_size, _ = self.get_dataloaders()

            # training process: train on progressive patchsize
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(int(self.num_iterations_per_epoch)):
                train_outputs.append(self.train_step(next(self.dataloader_train_cur_patch_size)))
            self.on_train_epoch_end(train_outputs)

            # validation process: evaluate current weights on final maximal patchsize
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val_max_patch_size)))
            self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()

        # test model on test split
        self.configuration_manager.set_patch_size(self.original_patch_size)
        self.configuration_manager.set_batch_size(self.original_batch_size)
        self.on_train_end()

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # original value of oversample_foreground_percent=0.33 is overwritten to 0.5, as nnU-Net falls back to a oversample_foreground_percent=0.5 for batchsize=2 (one forced foreground patch and one random/background patch)
        # to ensure a smooth class balance trajectory, we keep it oversample_foreground_percent=0.5 throughout the whole training
        self.oversample_foreground_percent = 0.5

        # this potentially can help handling very large numbers of files opened. could be helpful for extremely large batch sizes (>1000)
        os.environ["nnUNet_keep_files_open"] = "True"

        super().__init__(plans, configuration, fold, dataset_json, device)

        self.original_patch_size = self.configuration_manager.patch_size
        self.original_batch_size = self.configuration_manager.batch_size

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch =  1
        self.num_epochs = 1000
        self.current_epoch = 0


class nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance_GridSampler_Jitter_1Percent(nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance_GridSampler_Jitter):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # original value of oversample_foreground_percent=0.33 is overwritten to 0.5, as nnU-Net falls back to a oversample_foreground_percent=0.5 for batchsize=2 (one forced foreground patch and one random/background patch)
        # to ensure a smooth class balance trajectory, we keep it oversample_foreground_percent=0.5 throughout the whole training
        self.oversample_foreground_percent = 0.5

        # this potentially can help handling very large numbers of files opened. could be helpful for extremely large batch sizes (>1000)
        os.environ["nnUNet_keep_files_open"] = "True"

        super().__init__(plans, configuration, fold, dataset_json, device)

        self.original_patch_size = self.configuration_manager.patch_size
        self.original_batch_size = self.configuration_manager.batch_size

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.num_iterations_per_epoch = 3
        self.num_val_iterations_per_epoch =  1
        self.num_epochs = 1000
        self.current_epoch = 0

class nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance_NoMirroring(nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance_GridSampler_Jitter):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

class nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance_PacingV2(nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance_GridSampler_Jitter):

    def compute_volume_proportional_pacing(self, patch_sizes, total_epochs=1000):
        """
        Calculates the number of epochs for each patch size proportional to its volume.
        Guarantees integer outputs, a minimum of 1 epoch per patch, and exactly matches total_epochs.
        
        Parameters:
        -----------
        patch_sizes : list of tuples/lists
            List of patch dimensions, e.g., [(64, 64, 64), (96, 96, 96), (128, 128, 128)]
        total_epochs : int
            Total number of epochs to distribute (default: 1000)
            
        Returns:
        --------
        epochs_per_patch : list of ints
            Discrete number of epochs assigned to each patch size.
        """
        k = len(patch_sizes)
        if k > total_epochs:
            raise ValueError(f"Number of patches ({k}) cannot exceed total_epochs ({total_epochs}) due to min=1 constraint.")
        
        # 1. Calculate the volume of each patch size
        volumes = np.array([np.prod(patch) for patch in patch_sizes], dtype=float)
        
        # 2. Allocate the minimum of 1 epoch to every patch size upfront
        allocated_epochs = np.ones(k, dtype=int)
        remaining_epochs = total_epochs - k
        
        # If no remaining epochs to distribute, we are done
        if remaining_epochs == 0:
            return allocated_epochs.tolist()
            
        # 3. Distribute the remaining epochs proportionally based on volume
        proportional_shares = (volumes / np.sum(volumes)) * remaining_epochs
        
        # Take the floor (integer part) of the shares
        floor_shares = np.floor(proportional_shares).astype(int)
        allocated_epochs += floor_shares
        
        # 4. Handle rounding remainders using the Largest Remainder Method
        # Find out how many epochs we are short of reaching exactly `total_epochs`
        current_total = np.sum(allocated_epochs)
        shortage = total_epochs - current_total
        
        if shortage > 0:
            # Get the fractional remainders that were cut off by np.floor
            remainders = proportional_shares - floor_shares
            
            # Sort indices by largest remainder descending
            descending_remainder_indices = np.argsort(remainders)[::-1]
            
            # Distribute 1 epoch to the patches with the highest remainders until shortage is resolved
            for i in range(shortage):
                idx = descending_remainder_indices[i]
                allocated_epochs[idx] += 1

        return allocated_epochs.tolist()



    def run_training(self):
        """
        ############################## PGPS-curricula comment ##############################
        training loop for Progressive Growing Of Patch Size:
        1. We start training on minimal possible patch size (depends on network architecture/num of pooling operations)
        2. We increase the patch size stepwise linearly during training using the minimal possible patch size steps (depends on network architecture/num of pooling operations)
        3. We use the final patch size as inference patch size

        Minimal Patch Size: minimal patch size that does result in 1x1x1 vector in bottleneck (depends on network architecture/num of pooling operations)
        Patch Size Steps: we take the minimal possible patch size update (only even integer shapes in all network stages are allowed in network processing)
        Maximal Patch Size: we take the same patch size as standard nnUNet planning (nnUNet maximizes the patch size that can fit with a batch size of 2 into a GPU memory of 11GB)
        """

        self.on_train_start()
        self.initialized = False

        self.print_to_log_file(
            '######################### Progressive Growing of Patchsize Training Config #########################')
        self.print_to_log_file('Original patch size: ' + str(self.configuration_manager.patch_size))
        self.print_to_log_file('Original batch size: ' + str(self.configuration_manager.batch_size))

        num_pool_per_axis = np.sum(np.array(self.configuration_manager.pool_op_kernel_sizes) == 2,
                                   axis=0).tolist()  # get number of pooling operations per axis

        self.print_to_log_file('num_pool_per_axis defines smallest processable patch size: ' + str(num_pool_per_axis))
        self.min_patch_size = np.array(
            [2 ** (num_pool_per_axis[0]), 2 ** (num_pool_per_axis[1]), 2 ** (num_pool_per_axis[2])])
        self.print_to_log_file(
            'Minimal possible patch size and minimal possible patch size increments: ' + str(self.min_patch_size))
        self.num_pool_per_axis = num_pool_per_axis
        self.stage = 0

        # plan patch size scheduler: which patch size increments
        self.plan_batch_size()
        self.min_patch_size = self.patch_sizes[0]


        # plan pacing function
        epochs_schedule = self.compute_volume_proportional_pacing(self.patch_sizes, total_epochs=self.num_epochs)
        print("--- Pacing Schedule ---")
        for patch, epochs in zip(self.patch_sizes, epochs_schedule):
            self.print_to_log_file(f"Patch Size {patch} -> Trained for {epochs} epochs")

        self.print_to_log_file('Minimal USED patch size: ' + str(self.min_patch_size))

        num_stages = np.ceil(
            (self.original_patch_size[0] - self.min_patch_size[0]) / (2 ** num_pool_per_axis[0])) + np.ceil(
            (self.original_patch_size[1] - self.min_patch_size[1]) / (2 ** num_pool_per_axis[1])) + np.ceil(
            (self.original_patch_size[2] - self.min_patch_size[2]) / (2 ** num_pool_per_axis[2])) + 1
        self.print_to_log_file('Number of different patchsize phases : ' + str(num_stages))
        self.print_to_log_file(
            '######################### Progressive Growing of Patchsize Training Config #########################')

        if self.current_epoch != 0:
            # case of continuing network training
            print("continue training: from epoch: " + str(self.current_epoch))

            for completed_epoch in range(1, self.current_epoch):
                if ((completed_epoch % int(self.num_epochs / num_stages)) == 0):
                    self.stage = self.stage + 1

            print("continue with stage: " + str(self.stage))
            # increase patch size in minimal possible steps, for each patchsize-phase same number of epochs

            self.patch_size = self.patch_sizes[self.stage]
            self.batch_size = self.batch_sizes[self.stage]

            self.configuration_manager.set_patch_size(self.patch_size)
            self.print_to_log_file('New Patch Size : ' + str(self.patch_size))

            self.configuration_manager.set_batch_size(self.batch_size)
            self.print_to_log_file('New Batch Size : ' + str(self.batch_size))
            self.dataloader_train_cur_patch_size, self.dataloader_val_max_patch_size = self.get_dataloaders()

            self.initialized = True
        else:
            print("Training from scratch")

        for epoch in range(self.current_epoch, self.num_epochs):

            # we update the patch size according to the pacing schedule, which is based on the volume of the patch sizes. This ensures that we train longer on smaller patch sizes and shorter on larger patch sizes, which can help with stability and convergence.
            
            if not self.initialized:
                # initialize training with smallest possible patchsize (depends on num of pooling operations)

                self.patch_size = self.min_patch_size
                self.print_to_log_file('Current patchsize : ' + str(self.patch_size))
                self.configuration_manager.set_patch_size(self.patch_size)
                self.batch_size = self.batch_sizes[self.stage]
                self.configuration_manager.set_batch_size(self.batch_size)
                self.print_to_log_file('Current batchsize : ' + str(self.batch_size))
                self.dataloader_train_cur_patch_size, _ = self.get_dataloaders()
                # Maximal Patch Size Validation
                self.patch_size = self.original_patch_size
                self.configuration_manager.set_patch_size(self.patch_size)
                self.batch_size = self.original_batch_size
                self.configuration_manager.set_batch_size(self.batch_size)
                self.dataloader_train_max_patch_size, self.dataloader_val_max_patch_size = self.get_dataloaders()
                self.initialized = True

                self.patch_size = self.min_patch_size

            elif epoch >= sum(epochs_schedule[:self.stage]):

                if not np.array_equal(self.patch_size, self.original_patch_size):
                    # if we are already at the maximal patch size, we don't want to increase it anymore

                    self.stage = self.stage + 1

                    if self.stage == len(self.patch_sizes):
                        self.stage = self.stage - 1

                    # increase patch size in minimal possible steps, for each patchsize-phase around same number of epochs (1000 epochs / number of possible patch sizes)
                    # final maximal patch size is trained for residuals (1000 epochs / number of possible patch sizes + 1000 epochs % number of possible patch sizes)

                    self.patch_size = self.patch_sizes[self.stage]
                    current_patch_size = self.patch_size
                    self.batch_size = self.batch_sizes[self.stage]

                    self.configuration_manager.set_patch_size(self.patch_size)
                    self.print_to_log_file("Starting a new patch size phase!")
                    self.print_to_log_file('New Patch Size : ' + str(self.patch_size))

                    self.configuration_manager.set_batch_size(self.batch_size)
                    self.print_to_log_file('New Batch Size : ' + str(self.batch_size))
                    self.dataloader_train_cur_patch_size, _ = self.get_dataloaders()

            # training process: train on progressive patchsize
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(int(self.num_iterations_per_epoch)):
                train_outputs.append(self.train_step(next(self.dataloader_train_cur_patch_size)))
            self.on_train_epoch_end(train_outputs)

            # validation process: evaluate current weights on final maximal patchsize
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val_max_patch_size)))
            self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()

        # test model on test split
        self.configuration_manager.set_patch_size(self.original_patch_size)
        self.configuration_manager.set_batch_size(self.original_batch_size)
        self.on_train_end()



