# How to use Progressive Growing of Patch Size for Training nnU-Net**

This is the official implementation of the Journal Article **Progressive Growing of Patch Size: Improved Convergence for Patch-based Medical Image Segmentation via Curriculum
Learning**, which is an extension of the MICCAI 2024 main conference paper **Progressive growing of patch size: Resource-efficient curriculum learning for dense prediction tasks** accesible [here](https://papers.miccai.org/miccai-2024/paper/2008_paper.pdf).

## Why should you use the Progressive Growing of Patch Size?

Curriculum learning can offer faster convergence for deep learning training. Therefore, it can be used to improve segmentation performance or reduce training runtime.
We implemented two different curriculum modes and tested them on 15 popular datasets (Medical Segmentation Decathlon, BTCV, AMOS22, KiTS23, TotalSegmentatorV2, ToothFairy2):
1. Performance Mode: this mode significantly outperforms standard fixed patch size training in all 15 tested datasets (Medical Segmentation Decathlon, BTCV, AMOS22, KiTS23, TotalSegmentatorV2, ToothFairy2)j by improving the overall Dice Score by a relative ~1.5%, while only ~90% of the original runtime
2. Efficiency Mode: this mode performs on par with fixed patch size training, while only needing ~45% of the original runtime 
  
## How to use Progressive Growing of Patch Size?

Currently only the 3d_fullres UNET_CONFIGURATION is supported. The code for the curricula can be found [here](../nnunetv2/training/nnUNetTrainer/variants/sampling)

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
