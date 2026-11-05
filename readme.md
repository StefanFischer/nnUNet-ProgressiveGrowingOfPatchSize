# This is the Repository for Progressive Growing of Patch Size implemened in nnU-Net
The curriculum learning strategy implemented for nnU-Net improves the training convergence and thus also the final segmentation performance.
Click [here](https://github.com/StefanFischer/nnUNet-ProgressiveGrowingOfPatchSize/blob/ProgressiveGrowingOfPatchSizeV2.6.2/documentation/how_to_use_PGPS_curricula.md) if you wish to train a nnU-Net instance via the Progressive Growing of Patch Size Curriculum.


# Welcome to the new nnU-Net!

Click [here](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) if you were looking for the old one instead.

Coming from V1? Check out the [TLDR Migration Guide](documentation/tldr_migration_guide_from_v1.md). Reading the rest of the documentation is still strongly recommended ;-)

## **2024-04-18 UPDATE: New residual encoder UNet presets available!**
Residual encoder UNet presets substantially improve segmentation performance.
They ship for a variety of GPU memory targets. It's all awesome stuff, promised! 
Read more :point_right: [here](documentation/resenc_presets.md) :point_left:

Also check out our [new paper](https://arxiv.org/pdf/2404.09556.pdf) on systematically benchmarking recent developments in medical image segmentation. You might be surprised!

# What is nnU-Net?
Image datasets are enormously diverse: image dimensionality (2D, 3D), modalities/input channels (RGB image, CT, MRI, microscopy, ...), 
image sizes, voxel sizes, class ratio, target structure properties and more change substantially between datasets. 
Traditionally, given a new problem, a tailored solution needs to be manually designed and optimized  - a process that 
is prone to errors, not scalable and where success is overwhelmingly determined by the skill of the experimenter. Even 
for experts, this process is anything but simple: there are not only many design choices and data properties that need to 
be considered, but they are also tightly interconnected, rendering reliable manual pipeline optimization all but impossible! 

![nnU-Net overview](documentation/assets/nnU-Net_overview.png)

## Start Here

- First-time setup: [Installation and setup](documentation/getting-started/installation-and-setup.md)
- First run on your own data: [Getting Started](documentation/getting-started/README.md)
- Task-oriented docs: [How-to Guides](documentation/how-to/README.md)
- Formats, commands, and configuration details: [Reference](documentation/reference/README.md)
- Concepts and rationale: [Explanation](documentation/explanation/README.md)

## Quick Install

Install PyTorch for your hardware first, then install nnU-Net:

```bash
pip install nnunetv2
```

For the full setup, including `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`, see [Installation and setup](documentation/getting-started/installation-and-setup.md).

## Documentation

Start with the [documentation home](documentation/README.md).

Useful entry points:

- New users: [Getting Started](documentation/getting-started/README.md)
- Dataset preparation: [Prepare a dataset](documentation/how-to/prepare-a-dataset.md)
- Training workflow: [Train models](documentation/how-to/train-models.md)
- Inference workflow: [Run inference](documentation/how-to/run-inference.md)
- Recommended residual encoder presets: [Residual Encoder Presets in nnU-Net](documentation/resenc_presets.md)
- Finetuning from nnssl self-supervised checkpoints: [Finetune from nnssl checkpoints](documentation/finetuning_from_nnssl_checkpoints.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

## Scope

nnU-Net is built for supervised semantic segmentation. It supports 2D and 3D data, arbitrary channel definitions, multiple image formats, and dataset-specific adaptation of preprocessing and network configuration.

It performs particularly well in training-from-scratch settings such as biomedical datasets, challenge datasets, and non-standard imaging problems where off-the-shelf natural-image pretrained models are often a poor fit.

For a concise overview of the design, see [How nnU-Net works](documentation/explanation/how-nnunet-works.md).

## Citation

Please cite the following paper when using nnU-Net:

```text
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature Methods, 18(2), 203-211.
```

Additional recent work on residual encoder presets and benchmarking:

- [nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation](https://arxiv.org/pdf/2404.09556.pdf)

## Project Notes

- nnU-Net v2 is a complete reimplementation of the original nnU-Net with improved code structure and extensibility.
- Not every dataset creates every configuration. For example, the cascade is only generated when the dataset characteristics justify it.
- Detailed historical changes are summarized in [What is different in v2?](documentation/changelog.md).

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de)
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
