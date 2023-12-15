# Image generation with Diffusion Model
HSE Deep Learning course homework.
During this homework I implemented Diffusion Model which generates simple images
See the original [task statement](https://github.com/puhsu/dl-hse/tree/main/week06-transformers/bhw01).

## Code organization
```shell
├── README.md             <- Top-level README.
├── requirements.txt      <- project requirements.
├── preprocess_data.py    <- preprocessing code.
├── train.py              <- train code.
├── test.py               <- test code.
│
└── src                   <- main code directory.
    ├── loss.py                     <- transformer loss  
    ├── model.py                    <- transformer model  
    ├── tinystories_dataset.py      <- TinyStories dataset, collate_fn 
    └── utils.py                    <- utils: constants, Tokenizer and its' functions, WandbWriter
```

## Installation
1. To install libraries run from the root directory
```shell
pip3 install -r requirements.txt
```
2. The dataset I use is [ArtBench](https://paperswithcode.com/dataset/artbench-10). Download the 256x256 version here: [ArtBench-10 on GitHub](https://github.com/liaopeiyuan/artbench/blob/main/README.md). Download the 32x32 version by the previous link or from the [Kaggle (only 32x32)](https://www.kaggle.com/datasets/alexanderliao/artbench10), the extended version [Kaggle (256x256 + AI gen)](https://www.kaggle.com/datasets/ravidussilva/real-ai-art). 
If you run `train.py`, the 32x32 version will be downloaded automatically.
4. Download my VAE checkpoint from the [google drive](TODO).


## Train
1. To setup wandb, if you have not used wandb before, run `wandb login` in bash.
2. To train run
```shell
python3 train.py
```

## Test
```shell
python3 test.py
```
`test.py` contains two arguments:
* `"-c", "--config", default="configs/test.json", type=str, help="config file path (default: configs/test.json)"`
* `"-p", "--checkpoint-path", default="checkpoint.pth", type=str, help="checkpoint path (default: checkpoint.pth)"`
* `"-t", "--temperature", default=0.6, type=float, help="sampling temperature (default: 0.6)"`

## Wandb 
1. [Wandb project](https://wandb.ai/tgritsaev/tiny_stories_dl2/overview?workspace=user-tgritsaev).
2. [Wandb report](https://wandb.ai/tgritsaev/dl-2-tinystories/reports/bhw-dl-2-HSE-course-tinystories--Vmlldzo2MTUzNzk4).

## Authors
* Timofei Gritsaev

## Credits
DCGAN implementation was taken from the [pytorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).