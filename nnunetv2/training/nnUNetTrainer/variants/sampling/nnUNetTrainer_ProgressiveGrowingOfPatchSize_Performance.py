import torch
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerBN import nnUNetTrainerBN
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
import inspect
import multiprocessing
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import time

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import Union, Tuple

from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


from typing import Union, Tuple

from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.utilities.label_handling.label_handling import LabelManager


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



import abc
class BasicTransform(abc.ABC):
    """
    Transforms are applied to each sample individually. The dataloader is responsible for collating, or we might consider a CollateTransform

    We expect (C, X, Y) or (C, X, Y, Z) shaped inputs for image and seg (yes seg can have more color channels)

    No idea what keypoint and bbox will look like, this is Michaels turf
    """
    def __init__(self):
        pass

    def __call__(self, **data_dict) -> dict:
        params = self.get_parameters(**data_dict)
        return self.apply(data_dict, **params)

    def apply(self, data_dict, **params):
        if data_dict.get('image') is not None:
            data_dict['image'] = self._apply_to_image(data_dict['image'], **params)

        if data_dict.get('regression_target') is not None:
            data_dict['regression_target'] = self._apply_to_regr_target(data_dict['regression_target'], **params)

        if data_dict.get('segmentation') is not None:
            data_dict['segmentation'] = self._apply_to_segmentation(data_dict['segmentation'], **params)

        if data_dict.get('keypoints') is not None:
            data_dict['keypoints'] = self._apply_to_keypoints(data_dict['keypoints'], **params)

        if data_dict.get('bbox') is not None:
            data_dict['bbox'] = self._apply_to_bbox(data_dict['bbox'], **params)

        return data_dict

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        pass

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        pass

    def _apply_to_keypoints(self, keypoints, **params):
        pass

    def _apply_to_bbox(self, bbox, **params):
        pass

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str

class SegOnlyTransform(BasicTransform):
    def apply(self, data_dict: dict, **params) -> dict:
        if data_dict.get('segmentation') is not None:
            data_dict['segmentation'] = self._apply_to_segmentation(data_dict['segmentation'], **params)
        return data_dict

class ConvertSegmentationToRegionsTransform2(SegOnlyTransform):
    def __init__(self, regions: Union[List, Tuple], channel_in_seg: int = 0):
        super().__init__()
        self.regions = [torch.Tensor(i) if not isinstance(i, int) else torch.Tensor([i]) for i in regions]
        self.channel_in_seg = channel_in_seg

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        num_regions = len(self.regions)
        region_output = torch.zeros((num_regions, *segmentation.shape[1:]), dtype=torch.bool, device=segmentation.device)
        for region_id, region_labels in enumerate(self.regions):
            if len(region_labels) == 1:
                region_output[region_id] = segmentation[self.channel_in_seg] == region_labels
            else:
                region_output[region_id] = torch.isin(segmentation[self.channel_in_seg], region_labels)
        # we return bool here and leave it to the loss function to cast it to whatever it needs. Transferring bool to
        # device followed by cast on device should be faster than having fp32 here and transferring that
        return region_output


class DownsampleSegForDSTransform3(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales: Union[List, Tuple],
                 order: int = 0, input_key: str = "seg",
                 output_key: str = "seg", axes: Tuple[int] = None):
        """
        Downscales data_dict[input_key] according to ds_scales. Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently
        """
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        start = time.time()
        if self.axes is None:
            axes = list(range(2, len(data_dict[self.input_key].shape)))
        else:
            axes = self.axes

        output = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * len(axes)
            else:
                assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                            f'for each axis) then the number of entried in that tuple (here ' \
                                            f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

            if all([i == 1 for i in s]):
                output.append(data_dict[self.input_key])
            else:
                new_shape = np.array(data_dict[self.input_key].shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                new_shape = np.round(new_shape).astype(int)
                """
                ############################## PGPS-curricula comment ##############################

                expensive nnUNet code:
                out_seg = np.zeros(new_shape, dtype=data_dict[self.input_key].dtype)
                for b in range(data_dict[self.input_key].shape[0]):
                    for c in range(data_dict[self.input_key].shape[1]):
                        out_seg[b, c] = resize_segmentation(data_dict[self.input_key][b, c], new_shape[2:], self.order)

                we replaced it by vectorized version from pytorch:
                """
                out_seg = torch.nn.functional.interpolate(torch.tensor(data_dict[self.input_key]).to(torch.float), scale_factor=s, mode = 'nearest-exact')

                output.append(out_seg.numpy())
        data_dict[self.output_key] = output
        time_diff = time.time() - start
        #print(time_diff)
        return data_dict



class nnUNetDataLoaderBase(DataLoader):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 original_batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False):
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)
        assert isinstance(data, nnUNetDataset), 'nnUNetDataLoaderBase only supports dictionaries as data'
        self.indices = list(data.keys())

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.original_batch_size = original_batch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple(label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        if self.batch_size % 2 == 1 and self.oversample_foreground_percent == 0.5:
            if sample_idx == int(((self.batch_size) / 2)):
                return np.random.uniform() < self.oversample_foreground_percent # in between index propabilistic oversampling
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        data, seg, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False, old_crop: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                print("self.annotated_classes_key")
                print(self.annotated_classes_key)

                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!')
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
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
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs



class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        start_time = time.time()
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        force_fg = False

        # create all crops from two patients > very quick, less overfitting
        for j, i in enumerate(selected_keys):
            # load patient

            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)

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
            """

            crops_per_volume = np.ceil(self.batch_size / self.original_batch_size)
            if j % crops_per_volume == 0:
                # load only every n-th volume to reduce dataloading runtime
                data_orig, seg_orig, properties_orig = self._data.load_case(i)
                # if old_crop we can keep old bbox
                old_crop = False
                # if new patient, we want to switch to load only foreground or only background now
                force_fg = not force_fg
            else:
                # here we keep on loading only foreground or background
                # if old_crop we can keep old bbox
                old_crop = True

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data_orig.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties_orig['class_locations'], old_crop=old_crop)

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data_orig.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data_orig[this_slice]

            this_slice = tuple([slice(0, seg_orig.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg_orig[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}



class nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance(nnUNetTrainer):


    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size=np.array(patch_size), rotation_for_DA=rotation_for_DA, deep_supervision_scales=deep_supervision_scales, mirror_axes=mirror_axes, do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
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

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)
        return mt_gen_train, mt_gen_val



    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        """
        here we updated the dataloaders. nnUNet originally loads one crop/patch per patient
        We load more crops per patient to prevent extremely costly dataloading. Furthermore, there can be overfitting for small datasets if (almost) all patients are loaded per iteration
        See Batching strategy experiment in Journal Extension
        """

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        self.dataset_tr = dataset_tr
        self.dataset_val = dataset_val
        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size, self.configuration_manager.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=False)
            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size, self.configuration_manager.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=False)
        return dl_tr, dl_val

    @staticmethod
    def get_validation_transforms(deep_supervision_scales: Union[List, Tuple],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None) -> AbstractTransform:
        """
        ############################## PGPS-curricula comment ##############################
        here we overwrite the DownsampleSegForDSTransform2 used for deep supervision which is slow for very large batches
        its vectorized version is DownsampleSegForDSTransform3
        """

        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(
                ConvertSegmentationToRegionsTransform2(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_training_transforms(self, patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        """
        ############################## PGPS-curricula comment ##############################
        here we overwrite the DownsampleSegForDSTransform2 used for deep supervision which is slow for very large batches
        its vectorized version is DownsampleSegForDSTransform3
        """


        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(
                ConvertSegmentationToRegionsTransform2(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms



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
        self.original_patch_size = self.configuration_manager.patch_size
        self.original_batch_size = self.configuration_manager.batch_size

        self.print_to_log_file('######################### Progressive Growing of Patchsize Training Config #########################')
        self.print_to_log_file('Original patch size: ' + str(self.configuration_manager.patch_size))
        self.print_to_log_file('Original batch size: ' + str(self.configuration_manager.batch_size))

        num_pool_per_axis = np.sum(np.array(self.configuration_manager.pool_op_kernel_sizes) == 2, axis=0).tolist() # get number of pooling operations per axis

        self.print_to_log_file('num_pool_per_axis defines smallest processable patch size: ' + str(num_pool_per_axis))
        self.min_patch_size = np.array([2**(num_pool_per_axis[0]), 2**(num_pool_per_axis[1]), 2**(num_pool_per_axis[2])])
        self.print_to_log_file('Minimal possible patch size and minimal possible patch size increments: ' + str(self.min_patch_size))
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
        self.print_to_log_file('######################### Progressive Growing of Patchsize Training Config #########################')


        if self.current_epoch != 0:
            # case of continuing network training
            print("continue training: from epoch: " + str(self.current_epoch))

            for completed_epoch in range(self.current_epoch):
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
                self.batch_size = 2
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
                self.dataloader_train_cur_patch_size,  _ = self.get_dataloaders()

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


    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):


        # original value of oversample_foreground_percent=0.33 is overwritten to 0.5, as nnU-Net falls back to a oversample_foreground_percent=0.5 for batchsize=2 (one forced foreground patch and one random/background patch)
        # to ensure a smooth class balance trajectory, we keep it oversample_foreground_percent=0.5 throughout the whole training
        self.oversample_foreground_percent = 0.5

        # this potentially can help handling very large numbers of files opened. could be helpful for extremely large batch sizes (>1000)
        os.environ["nnUNet_keep_files_open"] ="True"

        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0


class nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance_NoMirroring(nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes