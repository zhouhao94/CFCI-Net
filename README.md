# RGBD Semantic Segmentation Method

![Framework](Framework.jpg?raw=true "framwork")

The official implementation of **CFCI-Net: Cross-modality Feature Calibration and Integration Network for RGB-D Semantic Segmentation (IEEE T-IV 2024)**:
More details can be found in our paper [[**paper**](https://ieeexplore.ieee.org/abstract/document/10634814)].


## Usage
### Installation
1. Requirements

- Python 3.7+
- PyTorch 1.7.0 or higher
- CUDA 10.2 or higher

2. Install all dependencies.
Install pytorch, cuda and cudnn, then install other dependencies via:
```shell
pip install -r requirements.txt
```

### Datasets

Orgnize the dataset folder in the following structure:
```shell
<datasets>
|-- <DatasetName1>
    |-- <RGBFolder>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- <DepthFolder>
        |-- <name1>.<ModalXFormat>
        |-- <name2>.<ModalXFormat>
        ...
    |-- <LabelFolder>
        |-- <name1>.<LabelFormat>
        |-- <name2>.<LabelFormat>
        ...
    |-- train.txt
    |-- test.txt
|-- <DatasetName2>
|-- ...
```

`train.txt` contains the names of items in training set, e.g.:
```shell
<name1>
<name2>
...
```

Our work uses HHA format as the input of depth information; the generation of HHA maps from Depth maps can refer to [https://github.com/charlesCXK/Depth2HHA-python](https://github.com/charlesCXK/Depth2HHA-python).

For preparation of other datasets, please refer to the original websites:
- [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [SUN-RGBD](https://rgbd.cs.princeton.edu/)
- [2D-3D-S](http://3Dsemantics.stanford.edu/)

### Train
1. Pretrain weights:

    Download the pretrained segformer and swin transformer here [pretrained segformer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5?usp=sharing).(Thanks for CMX's sharing of backbone weights)

3. Config

    Edit config file in `configs.py`, including dataset and network settings.

4. Run multi GPU distributed training:
    ```shell
    $ CUDA_VISIBLE_DEVICES="GPU IDs" python -m torch.distributed.launch --nproc_per_node="GPU numbers you want to use" train.py
    ```

- The tensorboard file is saved in `log_<datasetName>_<backboneSize>/tb/` directory.
- Checkpoints are stored in `log_<datasetName>_<backboneSize>/checkpoints/` directory.

### Evaluation
Run the evaluation by:
```shell
CUDA_VISIBLE_DEVICES="GPU IDs" python eval.py -d="Device ID" -e="epoch number or range"
```
If you want to use multi GPUs please specify multiple Device IDs (0,1,2...).


## Result
We offer the test results of the proposed method on different RGBD datasets:

### NYU-V2(40 categories)
| Architecture | Backbone | mIOU(SS) | mIOU(MS & Flip) |
|:---:|:---:|:---:|:---:|
| CFCI-Net (ResNet) | ResNet-101 | 53.5% | |
| CFCI-Net (SegFormer) | MiT-B2 | 54.2% | |
| CFCI-Net (SegFormer) | MiT-B3 | 55.6% | |
| CFCI-Net (SegFormer) | MiT-B4 | 56.4% | 56.6% |
| CFCI-Net (SegFormer) | MiT-B5 | 56.9% | 57.3% |
| CFCI-Net (Swin Transformer) | Swin-B | 56.9% | 57.1% |
| CFCI-Net (Swin Transformer) | Swin-L | 56.9% | 57.6% |

### SUN RGB-D(37 categories)
| Architecture | Backbone | mIOU(SS) |
|:---:|:---:|:---:|
| CFCI-Net (ResNet) | ResNet-101 | 50.8% |
| CFCI-Net (SegFormer) | MiT-B2 | 51.6% |
| CFCI-Net (SegFormer) | MiT-B3 | 52.7% |

### 2D-3D-S(13 categories)
| Architecture | Backbone | mIOU(SS) |
|:---:|:---:|:---:|
| CFCI-Net (ResNet) | ResNet-101 | 61.2% |
| CFCI-Net (SegFormer) | MiT-B2 | 62.6% |

## Notice


## Publication
If you find this repo useful, please consider referencing the following paper:
```
@article{zhou2024cfci,
  title={CFCI-Net: Cross-modality Feature Calibration and Integration Network for RGB-D Semantic Segmentation},
  author={Zhou, Hao and Yang, Xu and Qi, Lu and Chen, Haojie and Huang, Hai and Qin, Hongde},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgement

Our code is heavily based on [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) and [SA-Gate](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch), thanks for their brilliant work!

