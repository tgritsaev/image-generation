# Image generation with Diffusion Model
HSE Deep Learning course homework.

During this homework I implemented DCGAN, CVAE and diffusion model (all models are tested except of diffusion model).

The original [task statement](https://github.com/puhsu/dl-hse/tree/main/week06-transformers/bhw01).

## Code organization
```shell
├── README.md                       <- Top-level README.
├── requirements.txt                <- project requirements.
├── train.py                        <- train code.
├── congigs/               
│   ├── cvae_mnist_train.json
│   ├── cvae_train.json
│   ├── cvae32x32_train.json
│   └── dcgan_train.json
│   
├── results/...               
│   
├── scripts/               
│   └── create_gif.ipynb
│
└── src/                            <- main code directory.
    ├── datasets/
    │   ├── artbench10_32x32.py
    │   ├── artbench10_256x256.py
    │   ├── base_dataset.py
    │   ├── cats_faces.py
    │   └── mnist_wrapper.py 
    │            
    ├── models/
    │   ├── cvae/
    │   │   └── cvae.py
    │   │   
    │   ├── dcgan/
    │   │   └── dcgan.py
    │   │   
    │   ├── ddpm/
    │   │   ├── diffusion.py
    │   │   └── unet.py
    │   ├── __init__.py                 
    │   └── base_model.py                 
    │   
    ├── trainers/
    │   ├── __init__.py                 
    │   ├── trainer.py                 
    │   └── gan_trainer.py                 
    │
    ├── utils/   
    │   ├── __init__.py
    │   └── utils.py               
    │   
    ├── __init__.py
    └── collate.py
```

## Installation
1. To install libraries run from the root directory
```shell
pip3 install -r requirements.txt
```
2. I used 2 datasets:
* [CatsFaces on Kaggle](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) for DCGAN.
* [MNIST](https://en.wikipedia.org/wiki/MNIST_database) and [ArtBench](https://paperswithcode.com/dataset/artbench-10) for CVAE. 
MNIST and ArtBench-32x32 will be downloaded automatically.
Download the ArtBench-256x256 version here: [ArtBench-10 on GitHub](https://github.com/liaopeiyuan/artbench/blob/main/README.md), the extended version [Kaggle (256x256 + AI gen)](https://www.kaggle.com/datasets/ravidussilva/real-ai-art). 

## Train
1. To setup wandb, if you have not used it before, run `wandb login` in bash.
2. To train run
```shell
python3 train.py -c config_path -w True
```

## DCGAN results
| Generated samples during training  | Final generated samples |
| :---: | :---: |
| ![](https://github.com/tgritsaev/image-generation/blob/main/results/dcgan.gif)  | ![](https://github.com/tgritsaev/image-generation/blob/main/results/final_dcgan.png)  |

## CVAE results
| Reconstructed samples during training  | Generated samples during training | Final generated samples |
| :---: | :---: | :---: |
| ![](https://github.com/tgritsaev/image-generation/blob/main/results/recontsructed_mnist_cvae.gif)  | ![](https://github.com/tgritsaev/image-generation/blob/main/results/generated_mnist_cvae.gif) | ![](https://github.com/tgritsaev/image-generation/blob/main/results/final_mnist_cvae.png)

## Report 
Report is on the `results/report_ru.pdf` (only Russian).

## Wandb 
1. [GAN wandb project](https://wandb.ai/tgritsaev/dl2-gan-generation?workspace=user-tgritsaev).
2. [CVAE wandb project](https://wandb.ai/tgritsaev/dl2-cvae-generation?workspace=user-tgritsaev).

## Authors
* Timofei Gritsaev

## Credits
DCGAN implementation was taken from the [pytorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

DDPM impementation was taken from the [DL-2 in HSE homework](https://github.com/puhsu/dl-hse/blob/main/week08-VAE-Diff/shw5/homework.ipynb), which is based on the [openai/guided-diffusion](https://github.com/openai/guided-diffusion). Unet implementation was taken from the [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).