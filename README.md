# Image generation with Diffusion Model
HSE Deep Learning course homework.

During this homework I implemented DCGAN, CVAE and diffusion model (all models are tested except of diffusion model).

See the original [task statement](https://github.com/puhsu/dl-hse/tree/main/week06-transformers/bhw01).

## Code organization
```shell
├── README.md             <- Top-level README.
├── requirements.txt      <- project requirements.
├── train.py              <- train code.
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
└── src/                  <- main code directory.
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
2. I use 2 datasets:
* [CatsFaces on Kaggle](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) for DCGAN.
* [ArtBench](https://paperswithcode.com/dataset/artbench-10) for CVAE. Download the 256x256 version here: [ArtBench-10 on GitHub](https://github.com/liaopeiyuan/artbench/blob/main/README.md). Download the 32x32 version by the previous link or from the [Kaggle (only 32x32)](https://www.kaggle.com/datasets/alexanderliao/artbench10), the extended version [Kaggle (256x256 + AI gen)](https://www.kaggle.com/datasets/ravidussilva/real-ai-art). 

## Train
1. To setup wandb, if you have not used it before, run `wandb login` in bash.
2. To train run
```shell
python3 train.py -c config_path -w True
```
The final DCGAN model was trained with 

## Results
| Generated samples  | Final samples |
| ------------- | ------------- |
| ![DCGAN](https://github.com/tgritsaev/image-generation/blob/main/results/dcgan.gif)  | ![](https://github.com/tgritsaev/image-generation/blob/main/results/final_dcgan.png)  |

## Wandb 
1. [Wandb project](https://wandb.ai/tgritsaev/tiny_stories_dl2/overview?workspace=user-tgritsaev).
2. [Wandb report](https://wandb.ai/tgritsaev/dl-2-tinystories/reports/bhw-dl-2-HSE-course-tinystories--Vmlldzo2MTUzNzk4).

## Authors
* Timofei Gritsaev

## Credits
DCGAN implementation was taken from the [pytorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).