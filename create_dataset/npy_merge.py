# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:43:35 2018

@author: KIMJIHAE
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import optimizers
from keras import backend as K
from os import environ
from importlib import reload

import os
import time
from keras.callbacks import Callback
import pandas as pd

def set_keras_backend(backend):
    if K.backend() != backend:
        environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


set_keras_backend('tensorflow')
K.image_data_format()



class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('Testing loss : %.2f%%' % loss)
        print('Testing acc : %.2f%%' % acc)



def recover_3darrays(emotions_dir, xnp, ynp, dataset):
    """Generates a single X, y arrays using all Numpy binary file.
      Args:
        emotions_dir: String path to folder with emotion folders.
      Returns:
        An array X with all 3D images on the dataset.
        An array y with the labels of all 3D images on the dataset.
    """

    # {"1": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "0": 6}

    
    if dataset == 'MMI' or dataset == 'CKP':
        labels = ['0','1','3','4','5','6','7']
    if dataset == 'AFEW':
        labels = ['0','1','3','4','5','6','7']
    if dataset == 'FERA':
        labels = ['0', '1', '4', '5', '6']
    print("labels : ", labels)  # [0,1,2,3,4,5,6,7]

    for index1, label in enumerate(labels):
        if index1 == 0:  # natural

            print("Recovering arrays for label", label)
            print("dir : ", emotions_dir + label + '/')
            
            for index2, npy in enumerate(
                    os.listdir(emotions_dir + label + '/')):
                im = np.load(emotions_dir + label + '/' + npy)
                
                if index1 == 0 and index2 == 0:
                    X = np.zeros((0, im.shape[0], im.shape[1], im.shape[2],
                                  im.shape[3]))
                    y = np.zeros((0, len(labels)))
                X = np.append(X, [im], axis=0)
                
                y_temp = [0] * len(labels)
                for index, lab in enumerate(labels):
                    if int(label) == int(lab):
                        y_temp[index] = 1.0
                        break
                y = np.append(y, [y_temp], axis=0)
        else:
            print("Recovering arrays for label", label) #1,3,4,5,6,7
            for index2, npy in enumerate(os.listdir(emotions_dir + label)):
                
                im = np.load(emotions_dir + label + '/' + npy)
                X = np.append(X, [im], axis=0)

                y_temp = [0] * len(labels)
                for index, lab in enumerate(labels):
                    if int(label) == int(lab):
                        y_temp[index] = 1.0
                        break
                y = np.append(y, [y_temp], axis=0)

    print("\nShape of X array:", X.shape)
    print("Shape of y array:", y.shape)

    np.save(xnp, X);
    np.save(ynp, y);

    return X, y


def make_npy(dataset, aug, tot, mo, smf, preprocess):

    if smf == 's':
        smf2 = 'slow'
    elif smf == 'm':
        smf2 = 'middle'
    elif smf == 'f':
        smf2 = 'fast'

    if dataset == 'CKP':
        dataset2 = 'ckp'
    elif dataset == 'MMI':
        dataset2 = 'mmi'
    elif dataset == 'MMI_All':
        dataset2 = 'mmiall'
    elif dataset == 'FERA':
        dataset2 = 'fera'
    elif dataset == 'AFEW':
        dataset2 = 'afew'

    if mo == 'Minimum':
        mo2 = 'minimum'
    elif mo == 'Overlapped':
        mo2 = 'over'

    preprocess2 = preprocess
    if preprocess == 'normlbp':
        preprocess2 = 'norm_lbp'
    elif preprocess == 'normnlbp':
        preprocess2 = 'norm_nlbp'
    elif preprocess == 'pre':
        preprocess2 = 'preprocessed'
    elif preprocess == 'norm':
        preprocess2 = 'normalized'
        
    main_path = '/home/sjpark/FER'

    if tot != 'None':
        file_path = '{}/FER_npy_dataset/{}_{}_{}/{}/{}'.format(main_path, dataset, mo2, aug, tot, preprocess2)
    else:
        file_path = '{}/FER_npy_dataset/{}_{}_{}/{}/{}'.format(main_path, dataset, mo2, aug, tot, preprocess2)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    xnp = '{}/{}_aug_{}_{}_{}_x.npy'.format(file_path, dataset2, mo2, smf, preprocess)
    ynp = '{}/{}_aug_{}_{}_{}_y.npy'.format(file_path, dataset2, mo2, smf, preprocess)

    if os.path.exists(xnp) and os.path.exists(ynp):
        print("already exist file names....\n")
    else:
        print("data loading with recover 3darrays....\n")
        if tot != 'None':
            X1, Y1 = recover_3darrays('{}/{}/{}_{}_{}/{}/{}/{}/'.format(main_path, dataset, dataset, tot, aug, mo, preprocess2, smf2), xnp, ynp, dataset)
        else:
            X1, Y1 = recover_3darrays('{}/{}/{}_{}/{}/{}/{}/'.format(main_path, dataset, dataset, aug, mo, preprocess2, smf2), xnp, ynp, dataset)


if __name__ == '__main__':
    # make_npy(dataset, aug, tot, mo, smf, preprocess):
    make_npy('AFEW', '2', 'Test', 'Minimum', 'f', 'nlbp')
    K.clear_session()