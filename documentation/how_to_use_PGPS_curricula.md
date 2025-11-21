# How to use Progressive Growing of Patch Size for Training nnU-Net

This is the official implementation of the Journal Article **Progressive Growing of Patch Size: Improved Convergence for Patch-based Medical Image Segmentation via Curriculum
Learning** accesible [here](https://arxiv.org/pdf/2510.23241), which is an extension of the MICCAI 2024 main conference paper **Progressive growing of patch size: Resource-efficient curriculum learning for dense prediction tasks** accesible [here](https://papers.miccai.org/miccai-2024/paper/2008_paper.pdf).

In the journal extension we fixed an issue for the performance mode, which now allows to fully utilize the GPU memory. The old default nnUNet batch creation can lead to overfitting when loading large number of patients to create patches from. Furthermore, now the InstanceNorm3D Layer is patched to allow one more patch size stage (minimal patch size). 

## Why should you use the Progressive Growing of Patch Size?

Curriculum learning can offer faster convergence for deep learning training. Therefore, it can be used to improve segmentation performance or reduce training runtime.
We implemented two different curriculum modes and tested them on 15 popular datasets (Medical Segmentation Decathlon, BTCV, AMOS22, KiTS23, TotalSegmentatorV2, ToothFairy2):
1. Performance Mode: this mode significantly outperforms standard fixed patch size training in all 15 tested datasets (Medical Segmentation Decathlon, BTCV, AMOS22, KiTS23, TotalSegmentatorV2, ToothFairy2) by improving the overall Dice Score by a relative ~1.5%, while only needing ~90% of the original runtime
2. Efficiency Mode: this mode performs on par with fixed patch size training, while only needing ~45% of the original runtime

The benchmarking on 15 popular and diverse segmentation tasks, shows improved training convergence and segmentation performance for PGPS-Performance and comparable segmentation performance for PGPS-Efficiency, while for both modes the training time is reduced substantially:
<p align="center">
  <img src="https://github.com/user-attachments/assets/e6a929b0-e462-4531-9bbf-4b9a41fb287d" alt="Image 1" width="45%">
  <img src="https://github.com/user-attachments/assets/c9112b46-bdca-44db-8cfc-f4dbec44171e" alt="Image 2" width="45%">
</p>
  
## How to use Progressive Growing of Patch Size

Currently only the 3d_fullres UNET_CONFIGURATION is supported. The code for the curricula can be found [here](../nnunetv2/training/nnUNetTrainer/variants/sampling).

### Using the default UNet Version

First you need to understand how to use the original nnU-Net version, which you can find an instruction [here](how_to_use_nnunet.md). 
For using the curriculum, you only need to specify the Progressive Growing of Patch Size nnU-Net Trainer variant, while planning and preprocessing is not affected:

For the curriculum performance mode:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance
```

and for the curriculum efficiency mode:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_ProgressiveGrowingOfPatchSize_Efficiency
```

### Using the bigger ResNet UNet Version

You can easily just specify the ResNetUNet version in form of the correct planner (here the ResEncUNetM version) and the curriculum performance mode. You can find the different ResEncUNet versions and how to preprocess/plan for this [here](resenc_presets.md):
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance -p nnUNetResEncUNetMPlans
```

### Using costum backbones

If you want to use another costum network, here are the aspects you need to adjust in the code:
- Implement your architecture
- Think about disabling deep supervision in the trainer
- Adjust the patch size planner of the Progressive Growing of Patch Size curriculum, by setting the minimal processable patch size that your architecture can handle

### CAREFUL: When directional defined classes are presented

Directional defined classes are often present in multi-organ segmentation tasks (BTCV, AMOS22, TotalSegmentator) or in Tooth segmentation (ToothFairy). They are defined as left/right, upper/lower, or front/back structures or even combinations of multiple directions/axis (BTCV/AMOS: left/right kidney, ToothFairy2: Upper Right Central Incisor). As PGPS curricula do have limited context in the starting phase of training, the curriculum struggles with such classes. This is mostly due to nnU-Net defaulty used mirror data augmentation. **Be Careful if datasets contain directional-defined classes and also try disabling mirroring!** This might also result in improved performance for default nnU-Net with fixed patch size training. For this scenario, there are special trainer files available that disable mirroring augmentation during training and test-time.

For the curriculum performance mode:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_ProgressiveGrowingOfPatchSize_Performance_NoMirroring
```

and for the curriculum efficiency mode:
```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainer_ProgressiveGrowingOfPatchSize_Efficiency_NoMirroring
```

