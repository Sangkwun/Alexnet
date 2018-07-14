import pandas as pd
import numpy as np
import tensorflow as tf

from PIL import Image
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split 
"""
    To.do
    - make class for dataset with batch size
    - the class need to split data into train and test

"""

def check_dataset():
    directory_path = 'resources/flowers/'
    classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    df = pd.DataFrame(columns=['flower', 'path'])

    for flower in classes:
        path = directory_path + flower
        files = [directory_path + flower + '/' + f for f in listdir(path) if isfile(join(path, f))]

        for image_path in files:
            df = df.append({'flower': flower, 'path': image_path}, ignore_index=True)

    encoded = pd.get_dummies(df['flower'])
    df =  pd.concat([df, encoded], axis=1)

    df.to_csv('resources/dataset.csv', sep=',')


def read_dataset(path='resources/dataset.csv'):
    df = pd.read_csv(path, sep=',', header=0, index_col=0)

    return df

def image_to_array(path):
    size = 227, 227
    img = Image.open(path)
    arr = np.array(img.resize(size))

    return np.divide(arr, 255)


def path_to_4dtensor(paths, batch_size, num_iter):

    start = batch_size * num_iter
    end = batch_size * (num_iter + 1)
    
    paths = paths[start:end]
    arr = [image_to_array(path) for path in paths]
    arr = np.stack(arr,axis=0)

    return arr