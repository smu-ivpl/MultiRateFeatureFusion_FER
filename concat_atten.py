# -*- coding: utf-8 -*-
"""
Concatenated Model with multi-DJSTN Models
"""

from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Convolution3D, Dense, Dropout, Flatten, MaxPooling3D, Layer, concatenate
from tensorflow.keras.layers import Input, Concatenate, Embedding, Reshape, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
import theano
import tensorflow.keras.backend as K
import tensorflow.keras
from keras_self_attention import SeqSelfAttention

# K.set_image_dim_ordering('tf')
keras.backend.image_data_format()
theano.config.optimizer = "None"


# DJSTN Model with 5 layers
def CNN_app1(weights_path=None, input_tensor=None, input_shape=None, classes=6):
    K.set_image_data_format("channels_last")

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    print("img_input size : ", img_input.shape)

    conv_1 = Convolution3D(
        64, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(img_input)
    maxpool_1 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_1)
    batchnorm_1 = BatchNormalization()(maxpool_1)

    conv_2 = Convolution3D(
        128, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_1)
    maxpool_2 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_2)
    batchnorm_2 = BatchNormalization()(maxpool_2)

    conv_3 = Convolution3D(
        256, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_2)
    maxpool_3 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_3)
    batchnorm_3 = BatchNormalization()(maxpool_3)

    conv_4 = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_3)
    maxpool_4 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_4)
    batchnorm_4 = BatchNormalization()(maxpool_4)

    conv_5 = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_4)
    maxpool_5 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_5)
    batchnorm_5 = BatchNormalization()(maxpool_5)

    flat = Flatten(name="flat11")(batchnorm_5)
    fc_1 = Dense(1024, activation="relu", name="fc_11")(flat)
    dropout_1 = Dropout(0.5, name="dropout_11")(fc_1)
    fc_2 = Dense(500, activation="relu", name="fc_21")(dropout_1)
    dropout_2 = Dropout(0.5, name="dropout_21")(fc_2)
    fc_3 = Dense(classes, activation="softmax", name="fc_31")(dropout_2)


    inputs = img_input
    model = Model(inputs, fc_2)

    if weights_path:
        # cnn.load_weights(weights_path)
        model = load_model(weights_path)

    return model


# DJSTN Model with 10 layers
def CNN_app2(weights_path=None, input_tensor=None, input_shape=None, classes=6):
    K.set_image_data_format("channels_last")

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    print("img_input size : ", img_input.shape)

    conv_1 = Convolution3D(
        64, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(img_input)
    maxpool_1 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_1)
    batchnorm_1 = BatchNormalization()(maxpool_1)

    conv_1a = Convolution3D(
        64, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_1)
    maxpool_1a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_1a)
    batchnorm_1a = BatchNormalization()(maxpool_1a)

    conv_2 = Convolution3D(
        128, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_1a)
    maxpool_2 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_2)
    batchnorm_2 = BatchNormalization()(maxpool_2)

    conv_2a = Convolution3D(
        128, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_2)
    maxpool_2a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_2a)
    batchnorm_2a = BatchNormalization()(maxpool_2a)

    conv_3 = Convolution3D(
        256, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_2a)
    maxpool_3 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_3)
    batchnorm_3 = BatchNormalization()(maxpool_3)

    conv_3a = Convolution3D(
        256, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_3)
    maxpool_3a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_3a)
    batchnorm_3a = BatchNormalization()(maxpool_3a)

    conv_4 = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_3a)
    maxpool_4 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_4)
    batchnorm_4 = BatchNormalization()(maxpool_4)

    conv_4a = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_4)
    maxpool_4a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_4a)
    batchnorm_4a = BatchNormalization()(maxpool_4a)

    conv_5 = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_4a)
    maxpool_5 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_5)
    batchnorm_5 = BatchNormalization()(maxpool_5)

    conv_5a = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_5)
    maxpool_5a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_5a)
    batchnorm_5a = BatchNormalization()(maxpool_5a)

    flat = Flatten(name="flat2")(batchnorm_5a)
    fc_1 = Dense(1024, activation="relu", name="fc_122")(flat)
    dropout_1 = Dropout(0.5, name="dropout_122")(fc_1)
    fc_2 = Dense(500, activation="relu", name="fc_222")(dropout_1)
    dropout_2 = Dropout(0.5, name="dropout_222")(fc_2)
    fc_3 = Dense(classes, activation="softmax", name="fc_322")(dropout_2)

    inputs = img_input
    model = Model(inputs, fc_2)

    if weights_path:
        # cnn.load_weights(weights_path)
        model = load_model(weights_path)

    return model


# DJSTN Model with 15 layers
def CNN_app3(weights_path=None, input_tensor=None, input_shape=None, classes=6):
    K.set_image_data_format("channels_last")

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    print("img_input size : ", img_input.shape)

    conv_1 = Convolution3D(
        64, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(img_input)
    maxpool_1 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_1)
    batchnorm_1 = BatchNormalization()(maxpool_1)

    conv_1a = Convolution3D(
        64, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_1)
    maxpool_1a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_1a)
    batchnorm_1a = BatchNormalization()(maxpool_1a)

    conv_1b = Convolution3D(
        64, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_1a)
    maxpool_1b = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_1b)
    batchnorm_1b = BatchNormalization()(maxpool_1b)

    conv_2 = Convolution3D(
        128, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_1b)
    maxpool_2 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_2)
    batchnorm_2 = BatchNormalization()(maxpool_2)

    conv_2a = Convolution3D(
        128, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_2)
    maxpool_2a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_2a)
    batchnorm_2a = BatchNormalization()(maxpool_2a)

    conv_2b = Convolution3D(
        128, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_2a)
    maxpool_2b = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_2b)
    batchnorm_2b = BatchNormalization()(maxpool_2b)

    conv_3 = Convolution3D(
        256, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_2b)
    maxpool_3 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_3)
    batchnorm_3 = BatchNormalization()(maxpool_3)

    conv_3a = Convolution3D(
        256, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_3)
    maxpool_3a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_3a)
    batchnorm_3a = BatchNormalization()(maxpool_3a)

    conv_3b = Convolution3D(
        256, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_3a)
    maxpool_3b = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_3b)
    batchnorm_3b = BatchNormalization()(maxpool_3b)

    conv_4 = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_3b)
    maxpool_4 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_4)
    batchnorm_4 = BatchNormalization()(maxpool_4)

    conv_4a = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_4)
    maxpool_4a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_4a)
    batchnorm_4a = BatchNormalization()(maxpool_4a)

    conv_4b = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_4a)
    maxpool_4b = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_4b)
    batchnorm_4b = BatchNormalization()(maxpool_4b)

    conv_5 = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_4b)
    maxpool_5 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_5)
    batchnorm_5 = BatchNormalization()(maxpool_5)

    conv_5a = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_5)
    maxpool_5a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_5a)
    batchnorm_5a = BatchNormalization()(maxpool_5a)

    conv_5b = Convolution3D(
        512, (5, 5, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        data_format="channels_last")(batchnorm_5a)
    maxpool_5b = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2),
        padding="same")(conv_5b)
    batchnorm_5b = BatchNormalization()(maxpool_5b)

    flat = Flatten(name="flat13")(batchnorm_5b)
    fc_1 = Dense(1024, activation="relu", name="fc_13")(flat)
    dropout_1 = Dropout(0.5, name="dropout_13")(fc_1)
    fc_2 = Dense(500, activation="relu", name="fc_23")(dropout_1)
    dropout_2 = Dropout(0.5, name="dropout_23")(fc_2)
    fc_3 = Dense(classes, activation="softmax", name="fc_33")(dropout_2)

    inputs = img_input
    model = Model(inputs, fc_2)

    if weights_path:
        # cnn.load_weights(weights_path)
        model = load_model(weights_path)

    return model


def CNN_concat(model1, model2, model3, jf_class):
    
    # Self-attention
    sf1 = Reshape((1, model1.output.shape[1]))(model1.output)
    sf1 = SeqSelfAttention()(sf1)

    sf2 = Reshape((1, model2.output.shape[1]))(model2.output)
    sf2 = SeqSelfAttention()(sf2)

    sf3 = Reshape((1, model3.output.shape[1]))(model3.output)
    sf3 = SeqSelfAttention()(sf3)
    
    # Concatenate the reinforced features
    img_input = concatenate([sf1, sf2, sf3], axis=2)
    
    # Joint Fusion Classifier
    fc_1 = Dense(1024, activation="relu", name="fc_1")(img_input)
    dropout_1 = Dropout(0.5, name="dropout_1")(fc_1)
    fc_2 = Dense(500, activation="relu", name="fc_2")(dropout_1)
    dropout_2 = Dropout(0.5, name="dropout_2")(fc_2)
    fc_3 = Dense(jf_class, activation="softmax", name="fc_3")(dropout_2)

    merged = Model(inputs=[model1.input,
                          model2.input,
                          model3.input],
                   outputs=[fc_3])

    return merged


def attModels(X1, X2, X3, jf_class):

    app_at1 = CNN_app1(input_shape=X1[0].shape, classes=jf_class)
    app_at2 = CNN_app2(input_shape=X2[0].shape, classes=jf_class)
    app_at3 = CNN_app3(input_shape=X3[0].shape, classes=jf_class)
    
    # Concatenate multi-models with multi-depths
    model_atten = CNN_concat(app_at1, app_at2, app_at3, jf_class)

    return model_atten