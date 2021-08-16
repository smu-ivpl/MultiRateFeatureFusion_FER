# -*- coding: utf-8 -*-
"""
Train and Test the multi-DJSTN Models with multi-layers
"""

import numpy as np
import time
import random
import os
import math
import keras
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import Callback
from keras import backend as K
# from tensorflow.python.keras import backend as K

import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class TestCallback(Callback):
    def __init__(self, test_data, test_acc_history, test_loss_history):
        self.test_data = test_data
        self.test_acc_history = test_acc_history
        self.test_loss_history = test_loss_history

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('Testing loss : %.2f%%' % loss)
        print('Testing acc : %.2f%%' % acc)

        test_acc_history = self.test_acc_history
        test_loss_history = self.test_loss_history
        test_acc_history.append(acc)
        test_loss_history.append(loss)

        return test_acc_history, test_loss_history


def train_test_valid_random(x, y, train_idxs, valid_idxs, test_idxs,
                            app=True):

    X_train = []
    Y_train = []
    X_train_d = []
    Y_train_d = []
    for i in range(len(train_idxs)):
        X_train.append(x[train_idxs[i]])
        Y_train.append(y[train_idxs[i]])

    X_test = []
    Y_test = []
    X_test_d = []
    Y_test_d = []
    for i in range(len(test_idxs)):
        X_test.append(x[test_idxs[i]])
        Y_test.append(y[test_idxs[i]])

    X_valid = []
    Y_valid = []
    X_valid_d = []
    Y_valid_d = []
    for i in range(len(valid_idxs)):
        X_valid.append(x[valid_idxs[i]])
        Y_valid.append(y[valid_idxs[i]])

    X_train = np.asarray(X_train)
    X_valid = np.asarray(X_valid)
    X_test = np.asarray(X_test)

    Y_train = np.asarray(Y_train)
    Y_valid = np.asarray(Y_valid)
    Y_test = np.asarray(Y_test)

    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid 


def train_test_valid_random_split(numbers, x, y, data, app=True): 
    # neu : number of neutral dataset
    # aug2 : number of how many times the nuetral dataset augmented
    # aug : number of how many time the other expression datasets augmented
    if data == 0 or data == 5:  # CKP
        aug2 = 2
        neu = 600
        aug = 14
    elif data == 1 or data == 6:  # MMI
        neu = 416
        aug2 = 2
        aug = 14
    elif data == 2 or data == 7:  # FERA
        neu = 0
        aug2 = 1
        aug = 14

    train_idxs = []
    test_idxs = []
    valid_idxs = []
    
    # split nuetral emotion dataset
    if neu:
        dataset_neu = []
        for i in range(int(neu / aug2)):
            dataset_neu.append(i)
        random.shuffle(dataset_neu)
        
        # 0.65 : 0.2 : 0.15 = train : test : val
        train_idx_neu = int(len(dataset_neu) * 0.65)
        test_idx_neu = int(len(dataset_neu) * 0.85)

        for i in range(train_idx_neu):
            for j in range(aug2):
                train_idxs.append(numbers[dataset_neu[i] * aug2 + j])

        for i in range(test_idx_neu - train_idx_neu):
            for j in range(aug2):
                test_idxs.append(numbers[dataset_neu[i + train_idx_neu] * aug2 + j])

        for i in range(len(dataset_neu) - test_idx_neu):
            for j in range(aug2):
                valid_idxs.append(numbers[dataset_neu[i + test_idx_neu] * aug2 + j])
    
    # split other emotion dataset
    dataset = []

    emotion_num = int((len(x) - neu) / aug)
    for i in range(emotion_num):
        dataset.append(i)

    random.shuffle(dataset)
    
    # 0.65 : 0.2 : 0.15 = train : test : val
    train_idx = int(len(dataset) * 0.65)
    test_idx = int(len(dataset) * 0.85)

    for i in range(train_idx):
        for j in range(aug):
            num = neu + dataset[i] * aug + j
            train_idxs.append(numbers[dataset[i] * aug + j + neu])

    for i in range(test_idx - train_idx):
        for j in range(aug):
            test_idxs.append(numbers[dataset[i + train_idx] * aug + j + neu])

    for i in range(len(dataset) - test_idx):
        for j in range(aug):
            valid_idxs.append(numbers[dataset[i + test_idx] * aug + j + neu])

    return train_idxs, valid_idxs, test_idxs 


def train_valid_random(x, y, train_idxs, valid_idxs, app=True):

    X_train = []
    Y_train = []
    for i in range(len(train_idxs)):
        X_train.append(x[train_idxs[i]])
        Y_train.append(y[train_idxs[i]])

    X_valid = []
    Y_valid = []
    for i in range(len(valid_idxs)):
        X_valid.append(x[valid_idxs[i]])
        Y_valid.append(y[valid_idxs[i]])

    X_train = np.asarray(X_train)
    X_valid = np.asarray(X_valid)

    Y_train = np.asarray(Y_train)
    Y_valid = np.asarray(Y_valid)

    return X_train, Y_train, X_valid, Y_valid
    

def train_valid_random_split(numbers, x, y, data, app=True): 
    # neu : number of neutral dataset
    # aug2 : number of how many times the nuetral dataset augmented
    # aug : number of how many time the other expression datasets augmented
    if data == 0 or data == 5:  # CKP
        aug2 = 2
        neu = 600
        aug = 14
    elif data == 1 or data == 6:  # MMI
        neu = 416
        aug2 = 2
        aug = 14
    elif data == 2 or data == 7:  # FERA
        neu = 0
        aug2 = 1
        aug = 14
    elif data == 3 or data == 8:  # AFEW
        neu = 0
        aug2 = 1
        aug = 4

    train_idxs = []
    valid_idxs = []

    # split nuetral emotion dataset
    if neu:
        dataset_neu = []
        for i in range(int(neu / aug2)):
            dataset_neu.append(i)
        random.shuffle(dataset_neu)
        
        # 0.8 : 0.2 = train : val
        train_idx_neu = int(len(dataset_neu) * 0.8)

        for i in range(train_idx_neu):
            for j in range(aug2):
                train_idxs.append(numbers[dataset_neu[i] * aug2 + j])

        for i in range(len(dataset_neu) - train_idx_neu):
            for j in range(aug2):
                valid_idxs.append(numbers[dataset_neu[i + train_idx_neu] * aug2 + j])

    # split other emotion datasets
    dataset = []

    emotion_num = int((len(x) - neu) / aug)
    for i in range(emotion_num):
        dataset.append(i)
    # for i in range(int(len(x) / aug)):
    #     dataset.append(i)
    random.shuffle(dataset)
    
    # 0.8 : 0.2 = train : val
    train_idx = int(len(dataset) * 0.8)

    for i in range(train_idx):
        for j in range(aug):
            num = neu + dataset[i] * aug + j
            train_idxs.append(numbers[dataset[i] * aug + j + neu])

    for i in range(len(dataset) - train_idx):
        for j in range(aug):
            valid_idxs.append(numbers[dataset[i + train_idx] * aug + j + neu])

    return train_idxs, valid_idxs 


def test_random(x, y, test_idxs, app=True):

    X_test = []
    Y_test = []
    for i in range(len(test_idxs)):
        X_test.append(x[test_idxs[i]])
        Y_test.append(y[test_idxs[i]])

    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    return X_test, Y_test


def hybrid_network(x, r):
    # x for dataset selection
    # r for iteration

    print(x, r)

    time = datetime.datetime.now()
    t = time.strftime('%Y%m%d-%H%M%S')


    from concat_atten import attModels

    if x == 0 or x == 5:  # ck+
        jf_class = 7
        emotions = ['neutral', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

        print('-----------------------------------------')
        print('-------------------CK+-------------------')
        print('-----------------------------------------')

    elif x == 1 or x == 6:  # mmi

        jf_class = 7
        emotions = ['neutral', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

        print('-----------------------------------------')
        print('-------------------MMI-------------------')
        print('-----------------------------------------')

    elif x == 2 or x == 7:  # fera

        jf_class = 5
        emotions = ['relief', 'anger', 'fear', 'joy', 'sadness']

        print('-----------------------------------------')
        print('------------------FERA-------------------')
        print('-----------------------------------------')

    elif x == 3 or x == 8:  # afew

        jf_class = 7
        emotions = ['neutral', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

        print('-----------------------------------------')
        print('------------------AFEW-------------------')
        print('-----------------------------------------')


    elif x == 4 or x == 9:  # All

        jf_class = 5
        emotions = ['relief', 'anger', 'fear', 'joy', 'sadness']

        print('-----------------------------------------')
        print('------------------ALL--------------------')
        print('-----------------------------------------')

    # num = []

    for it in range(r):

        now = datetime.datetime.now()

        dataset = ['ckp_min', 'mmi_min', 'fera_min', 'afew_min', 'all_min', 'ckp_over', 'mmi_over', 'fera_over',
                   'afew_over', 'all_over']

        if x <= 4:
            mo = 'Minimum'
            mo2 = 'minimum'
        else:
            mo = 'Overlapped'
            mo2 = 'over'

        dataset1 = ['CKP', 'MMI', 'FERA', 'AFEW', 'ALL', 'CKP', 'MMI', 'FERA', 'AFEW', 'ALL']
        dataset2 = ['ckp', 'mmi', 'fera', 'afew', 'all', 'ckp', 'mmi', 'fera', 'afew', 'all']
        pre = ['pre', 'lbp', 'norm', 'normlbp']
        pre2 = ['preprocessed', 'lbp', 'normalized', 'norm_lbp']

        modeldir = 'weights/{}/{}/{}/'.format(dataset[x], t, pre[it])
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        save_dir = 'graph/{}/{}/{}'.format(dataset[x], t, pre[it])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if x == 3 or x == 8:
            # tv(train+validation) dataset
            appx1_tv = np.load(
                'FER_npy_dataset/{}_{}/Train/{}/new_{}_aug_{}_s_{}_x.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x], mo2,
                                                                              pre[it]))
            appx2_tv = np.load(
                'FER_npy_dataset/{}_{}/Train/{}/new_{}_aug_{}_f_{}_x.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x], mo2,
                                                                              pre[it]))
            appx3_tv = np.load(
                'FER_npy_dataset/{}_{}/Train/{}/new_{}_aug_{}_m_{}_x.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x], mo2,
                                                                              pre[it]))

            appy1_tv = np.load(
                'FER_npy_dataset/{}_{}/Train/{}/new_{}_aug_{}_s_{}_y.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x], mo2,
                                                                              pre[it]))
            appy2_tv = np.load(
                'FER_npy_dataset/{}_{}/Train/{}/new_{}_aug_{}_f_{}_y.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x], mo2,
                                                                              pre[it]))
            appy3_tv = np.load(
                'FER_npy_dataset/{}_{}/Train/{}/new_{}_aug_{}_m_{}_y.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x], mo2,
                                                                              pre[it]))

            # test dataset
            appx1_t = np.load(
                'FER_npy_dataset/{}_{}/Test/{}/new_{}_aug_{}_s_{}_x.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                             mo2,
                                                                             pre[it]))
            appx2_t = np.load(
                'FER_npy_dataset/{}_{}/Test/{}/new_{}_aug_{}_f_{}_x.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                             mo2,
                                                                             pre[it]))
            appx3_t = np.load(
                'FER_npy_dataset/{}_{}/Test/{}/new_{}_aug_{}_m_{}_x.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                             mo2,
                                                                             pre[it]))

            appy1_t = np.load(
                'FER_npy_dataset/{}_{}/Test/{}/new_{}_aug_{}_s_{}_y.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                             mo2,
                                                                             pre[it]))
            appy2_t = np.load(
                'FER_npy_dataset/{}_{}/Test/{}/new_{}_aug_{}_f_{}_y.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                             mo2,
                                                                             pre[it]))
            appy3_t = np.load(
                'FER_npy_dataset/{}_{}/Test/{}/new_{}_aug_{}_m_{}_y.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                             mo2,
                                                                             pre[it]))
        else:
            appx1 = np.load(
                'FER_npy_dataset/{}_{}_14/ALL/{}/{}_aug_{}_s_{}_x.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                              mo2,
                                                                              pre[it]))
            appx2 = np.load(
                'FER_npy_dataset/{}_{}_14/ALL/{}/{}_aug_{}_f_{}_x.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                              mo2,
                                                                              pre[it]))
            appx3 = np.load(
                'FER_npy_dataset/{}_{}_14/ALL/{}/{}_aug_{}_m_{}_x.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                              mo2,
                                                                              pre[it]))

            appy1 = np.load(
                'FER_npy_dataset/{}_{}_14/ALL/{}/{}_aug_{}_s_{}_y.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                              mo2,
                                                                              pre[it]))
            appy2 = np.load(
                'FER_npy_dataset/{}_{}_14/ALL/{}/{}_aug_{}_f_{}_y.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                              mo2,
                                                                              pre[it]))
            appy3 = np.load(
                'FER_npy_dataset/{}_{}_14/ALL/{}/{}_aug_{}_m_{}_y.npy'.format(dataset1[x], mo2, pre2[it], dataset2[x],
                                                                              mo2,
                                                                              pre[it]))

        if pre[it] == 'pre':  # pre

            print('-----------------------------------------')
            print('-------------------PRE-------------------')
            print('-----------------------------------------')

        elif pre[it] == 'lbp':  # lbp

            print('-----------------------------------------')
            print('-------------------LBP-------------------')
            print('-----------------------------------------')

        elif pre[it] == 'norm':  # norm

            print('-----------------------------------------')
            print('------------------NORM-------------------')
            print('-----------------------------------------')

        elif pre[it] == 'normlbp':  # norm_lbp

            print('-----------------------------------------')
            print('------------------nLBP-------------------')
            print('-----------------------------------------')

        if x == 3 or x == 8:
            # random index shuffle for train_test_valid split
            numbers_tv = []
            for i in range(len(appy1_tv)):
                numbers_tv.append(i)

            # random index shuffle for train_test_valid split
            numbers_test = []
            for i in range(len(appy1_t)):
                numbers_test.append(i)

        else:
            # random index shuffle for train_test_valid split
            numbers = []
            for i in range(len(appy1)):
                numbers.append(i)


        if it == 0:
            if x == 3 or x == 8:
                train_idxs, valid_idxs = train_valid_random_split(numbers_tv, appx1_tv, appy1_tv, x, True)
                test_idxs = numbers_test[:len(appy1_tv)]
                random.shuffle(test_idxs)
                random.shuffle(train_idxs)
                random.shuffle(valid_idxs)
            else:
                train_idxs, valid_idxs, test_idxs = train_test_valid_random_split(numbers, appx1, appy1, x, True)
                random.shuffle(train_idxs)
                random.shuffle(test_idxs)
                random.shuffle(valid_idxs)


        # idx not fixed
        if x == 3 or x == 8:
            # slow
            X_app_train1, Y_app_train1, X_app_valid1, Y_app_valid1 = train_valid_random(
                appx1_tv, appy1_tv, train_idxs, valid_idxs, True)
            # fast
            X_app_train2, Y_app_train2, X_app_valid2, Y_app_valid2 = train_valid_random(
                appx2_tv, appy2_tv, train_idxs, valid_idxs, True)
            # middle
            X_app_train3, Y_app_train3, X_app_valid3, Y_app_valid3 = train_valid_random(
                appx3_tv, appy3_tv, train_idxs, valid_idxs, True)

            # slow
            X_app_test1, Y_app_test1 = test_random(appx1_t, appy1_t, test_idxs, True)
            # fast
            X_app_test2, Y_app_test2 = test_random(appx2_t, appy2_t, test_idxs, True)
            # middle
            X_app_test3, Y_app_test3 = test_random(appx3_t, appy3_t, test_idxs, True)

        else:
            # slow
            X_app_train1, Y_app_train1, X_app_test1, Y_app_test1, X_app_valid1, Y_app_valid1 = train_test_valid_random(
                appx1, appy1, train_idxs, valid_idxs, test_idxs, True)
            # fast
            X_app_train2, Y_app_train2, X_app_test2, Y_app_test2, X_app_valid2, Y_app_valid2 = train_test_valid_random(
                appx2, appy2, train_idxs, valid_idxs, test_idxs, True)
            # middle
            X_app_train3, Y_app_train3, X_app_test3, Y_app_test3, X_app_valid3, Y_app_valid3 = train_test_valid_random(
                appx3, appy3, train_idxs, valid_idxs, test_idxs, True)

        import csv

        test_idxs_s = sorted(test_idxs)
        with open('{}/test_indexs.csv'.format(save_dir), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(test_idxs_s)


        ep = 150  # epoch
        batch = 32

        adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        
        #################################################
        print("++++++++++++++++++++++++++++")
        print("++++++++++DATA Concat+++++++")
        print("++++++++++++++++++++++++++++")
        
        print("++++++++++++++++++++++++++++")
        print("+++++++++++++ATT++++++++++++")
        print("++++++++++++++++++++++++++++")

        # attention model
        model_atten = attModels(X_app_train1, X_app_train3, X_app_train2, jf_class)

        modelweight_name = 'concat_atten_network.h5'
        modelweight_path = os.path.join(modeldir, modelweight_name)

        cp_callback = ModelCheckpoint(modelweight_path,
                                      monitor='val_accuracy', mode='max',
                                      save_best_only=True,
                                      save_weights_only=True,
                                      verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        early_stopping2 = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100, baseline=1)


        model_atten.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])
        # model_atten.summary()
        Y_app_train1_1 = np.expand_dims(Y_app_train1, axis=1)
        Y_app_valid1_1 = np.expand_dims(Y_app_valid1, axis=1)
        app_history1 = model_atten.fit([X_app_train1, X_app_train3, X_app_train2],
                                       Y_app_train1_1,
                                       batch_size=batch,
                                       validation_data=([X_app_valid1, X_app_valid3, X_app_valid2], Y_app_valid1_1),
                                       epochs=ep,
                                       shuffle=True,
                                       verbose=1,
                                       callbacks=[cp_callback])

        # without weighted model
        # model_soft.load_weights(modelweight_path)
        Y_app_test1_1 = np.expand_dims(Y_app_test1, axis=1)
        scores1 = model_atten.evaluate([X_app_test1, X_app_test3, X_app_test2], Y_app_test1_1)
        appacc1 = scores1[1] * 100
        print("Test Accuracy: %.2f%%" % (scores1[1] * 100))
        print("Test Loss:%.2f%%" % (scores1[0]))

        # if you want to show every figure, use plt.show()
        # accuracy graph of train and validation set
        plt.figure(figsize=[8, 6])
        plt.plot(app_history1.history['accuracy'], 'r', linewidth=3.0)
        plt.plot(app_history1.history['val_accuracy'], 'b', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        plt.xlabel('Iteration ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves', fontsize=16)
        plt.savefig('{}/{}_without_concat_atten_accuracy_{}.png'.format(save_dir, it, appacc1))
        # plt.show()
        plt.close()

        # loss graph of train and validation set
        plt.figure(figsize=[8, 6])
        plt.plot(app_history1.history['loss'], 'r', linewidth=3.0)
        plt.plot(app_history1.history['val_loss'], 'b', linewidth=3.0)
        plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('loss curves')
        plt.savefig('{}/{}_without_concat_atten_loss_{}.png'.format(save_dir, it, appacc1))
        # plt.show()
        plt.close()


        # with weighted model
        model_atten.load_weights(modelweight_path)
        scores1 = model_atten.evaluate([X_app_test1, X_app_test3, X_app_test2], Y_app_test1_1)
        appacc1 = scores1[1] * 100
        print("Test Accuracy: %.2f%%" % (scores1[1] * 100))
        print("Test Loss:%.2f%%" % (scores1[0]))

        # if you want to show every figure, use plt.show()
        # accuracy graph of train and validation set
        plt.figure(figsize=[8, 6])
        plt.plot(app_history1.history['accuracy'], 'r', linewidth=3.0)
        plt.plot(app_history1.history['val_accuracy'], 'b', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves', fontsize=16)
        plt.savefig('{}/{}_concat_atten_accuracy_{}.png'.format(save_dir, it, appacc1))
        # plt.show()
        plt.close()

        # loss graph of train and validation set
        plt.figure(figsize=[8, 6])
        plt.plot(app_history1.history['loss'], 'r', linewidth=3.0)
        plt.plot(app_history1.history['val_loss'], 'b', linewidth=3.0)
        plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('loss curves')
        plt.savefig('{}/{}_concat_atten_loss_{}.png'.format(save_dir, it, appacc1))
        # plt.show()
        plt.close()

        K.clear_session()
      
        
        # clear datasets list
        X_app_train1 = None
        Y_app_train1 = None
        X_app_test1 = None
        Y_app_test1 = None
        X_app_valid1 = None
        Y_app_valid1 = None

        X_app_train2 = None
        Y_app_train2 = None
        X_app_test2 = None
        Y_app_test2 = None
        X_app_valid2 = None
        Y_app_valid2 = None

        X_app_train3 = None
        Y_app_train3 = None
        X_app_test3 = None
        Y_app_test3 = None
        X_app_valid3 = None
        Y_app_valid3 = None

        appx1 = None
        appy1 = None

        appx2 = None
        appy2 = None

        appx1 = None
        appy1 = None

    K.clear_session()


if __name__ == '__main__':
    print('-----------------------------------------')
    print('>>          FER TRAIN SYSTEM            <<')
    print('-----------------------------------------')
    print('-----------SELECT FER DATASET------------')
    print('> (0) : CK+_minimum')
    print('> (1) : MMI_minimum')
    print('> (2) : GEMEP-FERA_minimum')
    print('> (3) : AFEW_minimum')
    print('> (4) : ALL_minimum')
    print('> (5) : CK+_over')
    print('> (6) : MMI_over')
    print('> (7) : GEMEP-FERA_over')
    print('> (8) : AFEW_over')
    print('> (9) : ALL_over')
    print('-----------------------------------------')
    x = int(input())  # selected dataset (0 : ck+, 1 : mmi, 2 : fera)
    print('----------THE NUMBER OF EPOCH------------')
    r = int(input())  # iteration of training&testing preprocessing list
    print('-----------------------------------------')
    hybrid_network(x, r)

    K.clear_session()
