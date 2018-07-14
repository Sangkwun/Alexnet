# Alexnet with Tensorflow.

This project is based on Alexnet.<br/>
Tensorflow is used to build alexnet network and used Pillow and numpy to load and convert dataset.<br/>

https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf<br/>



## Requirements

This project used pipenv so you can check pipfile in repository. <br/>
you can just install all requirements with pipenv install. <br/>

- numpy: 1.14.5
- pandas: 0.23.1
- pillow: 5.1.0
- tensorflow: 1.8.0


## Dataset

For training alexnet flower dataset used. There are 4242 images about 5 classes of flowers. <br/>
It's divided in directory of flower names so easy to handle it.

- https://www.kaggle.com/alxmamaev/flowers-recognition

## Training

You can train model with train.py.
Before execute train you have to prepare dataset.
So download it into resources then unzip it.
training_dataset.csv and valid_dataset, etc are splitted dataset with ratio.
train.py train model with above csv file.

- you can change dropout rate and when model is creating.
- tensorboard log will be saved in logdir
- weights will be saved in weights

![alt text](https://github.com/Sangkwun/Alexnet/blob/master/accuracy.png)

![alt text](https://github.com/Sangkwun/Alexnet/blob/master/cost.png)

## Inference

It doesn't have command yet.
so modify image_path in inference.py then execute it.

- first output: [[ -9.98178387 -14.03773689   8.21770668  -0.54801857  -3.23122072]]
- index of class: 2
- The class of image is rose
