# You Need to Read Again: Multi-granularity Perception Network for Moment Retrieval in Videos

## Introduction

This is an implementation repository for our work in SIGIR 2022.
**You Need to Read Again: Multi-granularity Perception Network for Moment Retrieval in Videos**. [paper](https://arxiv.org/pdf/2205.12886.pdf)

![](https://github.com/Huntersxsx/MGPN/blob/master/img/framework.png)

<!-- ## Note:
Our pre-trained models are available at [SJTU jbox](https://jbox.sjtu.edu.cn/l/215Z2T) or [baiduyun, passcode:xmc0](https://pan.baidu.com/s/1CRojAlDURJ57tUprdNbfFg) or [Google Drive](https://drive.google.com/drive/folders/1AFdgfxFCA9ji36HaveL2dQ7wr7OjlHjb?usp=sharing). -->


## Installation

Clone the repository and move to folder:
```bash
git clone https://github.com/Huntersxsx/MGPN.git

cd MGPN
```

To use this source code, you need Python3.7+ and a few python3 packages:
- pytorch 1.1.0
- torchvision 0.3.0
- torchtext
- easydict
- terminaltables
- tqdm

## Data
We use the data offered by [2D-TAN](https://github.com/microsoft/2D-TAN/tree/master/data), and the extracted features can be found at [Box](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav/folder/137471266949).

</br>

The folder structure should be as follows:
```
.
├── checkpoints
│   ├── best
│   │    ├── TACoS
│   │    ├── ActivityNet
│   │    └── Charades
├── data
│   ├── TACoS
│   │    ├── tall_c3d_features.hdf5
│   │    └── ...
│   ├── ActivityNet
│   │    ├── sub_activitynet_v1-3.c3d.hdf5
│   │    └── ...
│   ├── Charades-STA
│   │    ├── charades_vgg_rgb.hdf5
│   │    └── ...
│
├── experiments
│
├── lib
│   ├── core
│   ├── datasets
│   └── models
│
└── moment_localization
```

## Train and Test
Please download the visual features from [Box](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav) and save it to the `data/` folder.

#### Training
Use the following commands for training:
- For TACoS dataset, run: 
```bash
    sh run_tacos.sh
```
- For ActivityNet-Captions dataset, run:
```bash
    sh run_activitynet.sh
```
- For Charades-STA dataset, run:
```bash
    sh run_charades.sh
```

<!-- #### Testing
Our trained model are provided in [SJTU jbox](https://jbox.sjtu.edu.cn/l/215Z2T) or [baiduyun, passcode:xmc0](https://pan.baidu.com/s/1CRojAlDURJ57tUprdNbfFg) or [Google Drive](https://drive.google.com/drive/folders/1AFdgfxFCA9ji36HaveL2dQ7wr7OjlHjb?usp=sharing). Please download them to the `checkpoints/best/` folder.
Use the following commands for testing:
- For TACoS dataset, run: 
```bash
    sh test_tacos.sh
```
- For ActivityNet-Captions dataset, run:
```bash
    sh test_activitynet.sh
```
- For Charades-STA dataset, run:
```bash
    sh test_charades.sh
``` -->

## Main results:

| **TACoS** | Rank1@0.3 | Rank1@0.5 | Rank5@0.3 | Rank5@0.5 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **MGPN<sub>256** |  48.81 | 36.74 |  71.46 | 59.24 |
</br>

| **ActivityNet** | Rank1@0.5 | Rank1@0.7 | Rank5@0.6 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **MGPN<sub>256** | 47.92 | 30.47 | 78.15 | 63.56 |
</br>

<!-- | **Charades (VGG)**  | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **RaNet** | 43.87 | 26.83 | 86.67 | 54.22 |
</br> -->

| **Charades (I3D)**  | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **MGPN<sub>256** | 60.82 | 41.16 | 89.77 | 64.73 |

## Acknowledgement

We greatly appreciate the [2D-Tan repository](https://github.com/microsoft/2D-TAN). Please remember to cite the papers:

```
@article{sun2022you,
  title={You Need to Read Again: Multi-granularity Perception Network for Moment Retrieval in Videos},
  author={Sun, Xin and Wang, Xuan and Gao, Jialin and Liu, Qiong and Zhou, Xi},
  journal={arXiv preprint arXiv:2205.12886},
  year={2022}
}

@inproceedings{gao2021relation,
  title={Relation-aware Video Reading Comprehension for Temporal Language Grounding},
  author={Gao, Jialin and Sun, Xin and Xu, Mengmeng and Zhou, Xi and Ghanem, Bernard},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={3978--3988},
  year={2021}
}

@InProceedings{2DTAN_2020_AAAI,
author = {Zhang, Songyang and Peng, Houwen and Fu, Jianlong and Luo, Jiebo},
title = {Learning 2D Temporal Adjacent Networks forMoment Localization with Natural Language},
booktitle = {AAAI},
year = {2020}
} 


```
