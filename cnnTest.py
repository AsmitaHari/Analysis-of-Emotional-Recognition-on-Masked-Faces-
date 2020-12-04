import os
import cv2
import math
import numpy as np
import glob
import csv
from shutil import copyfile,copy
from collections import OrderedDict
import scikitplot

import seaborn as sns
from matplotlib import pyplot

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

from keras import backend as K
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
labelValues = {0:'neutral',1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy',6:'sad',7:'surprise'}
TOP_EMOTIONS = ["neutral","happy", "surprise", "anger", "sadness", "fear"]
INPUT_PATH = "newOutFolder/"

def createEmotionFolder():
    if not os.path.isdir('newOutFolder'):
        os.mkdir('newOutFolder')

    with open('imageFolder.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for dat in csv_reader:
            checkLen = False
            path = os.path.join(dat[0],"*.png")
            count = 0
            for imagePath in glob.glob(path):
                dirName = os.path.dirname(imagePath)
                length = len(glob.glob(dirName + "/*.png"))
                count+=1
                filepath = "newOutFolder/"+labelValues[int(dat[1])]
                if count == length:
                    if not os.path.isdir(filepath):
                        os.mkdir(filepath)
                    copy(imagePath,filepath)
    print("Done with files")

def loadImages(total_images):
    img_arr = np.empty(shape=(total_images, 48, 48, 1))
    img_label = np.empty(shape=(total_images))
    label_to_text = {}

    idx = 0
    label = 0
    for dir_ in os.listdir(INPUT_PATH):
        if dir_ in TOP_EMOTIONS:
            for f in os.listdir(INPUT_PATH + dir_ + "/"):
                img = cv2.imread(INPUT_PATH + dir_ + "/" + f, 0)
                resized = cv2.resize(img, (48,48), interpolation=cv2.INTER_AREA)
                img_arr[idx] = np.expand_dims(resized, axis=2)
                img_label[idx] = label
                idx += 1
            label_to_text[label] = dir_
            label += 1

    img_label = np_utils.to_categorical(img_label)

    print(img_arr.shape, img_label.shape, label_to_text)

    X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label, train_size=0.7, stratify=img_label,
                                                        shuffle=True, random_state=42)
    print(X_train.shape, X_test.shape)

    fig = pyplot.figure(1, (10, 10))

    idx = 0
    for k in label_to_text:
        sample_indices = np.random.choice(np.where(y_train[:, k] == 1)[0], size=5, replace=False)
        sample_images = X_train[sample_indices]
        for img in sample_images:
            idx += 1
            ax = pyplot.subplot(5, 5, idx)
            ax.imshow(img.reshape(48, 48), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label_to_text[k])
            pyplot.tight_layout()
    optim = optimizers.Adam(0.001)

    model = build_dcnn(input_shape=(48, 48, 1), num_classes=len(label_to_text))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00008,
        patience=12,
        verbose=1,
        restore_best_weights=True,
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.4,
        patience=6,
        min_lr=1e-7,
        verbose=1,
    )

    callbacks = [
        early_stopping,
        lr_scheduler,
    ]

    batch_size = 10
    epochs = 60
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    train_datagen.fit(X_train)

    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        callbacks=callbacks,
        use_multiprocessing=True
    )
    sns.set()
    fig = pyplot.figure(0, (12, 4))

    ax = pyplot.subplot(1, 2, 1)
    sns.lineplot(history.epoch, history.history['accuracy'], label='train')
    sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
    pyplot.title('Accuracy')
    pyplot.tight_layout()

    ax = pyplot.subplot(1, 2, 2)
    sns.lineplot(history.epoch, history.history['loss'], label='train')
    sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
    pyplot.title('Loss')
    pyplot.tight_layout()

    pyplot.savefig('epoch_history.png')
    pyplot.show()

    text_to_label = dict((v, k) for k, v in label_to_text.items())
    text_to_label

    yhat_test = model.predict(X_test)
    yhat_test = np.argmax(yhat_test, axis=1)
    ytest_ = np.argmax(y_test, axis=1)

    scikitplot.metrics.plot_confusion_matrix(ytest_, yhat_test, figsize=(7, 7))
    pyplot.savefig("confusion_matrix_model3pipes.png")

    test_accu = np.sum(ytest_ == yhat_test) / len(ytest_) * 100
    print(f"test accuracy: {round(test_accu, 4)} %\n\n")

    print(classification_report(ytest_, yhat_test))


def build_dcnn(input_shape, num_classes):
    model_in = Input(shape=input_shape, name="input")

    conv2d_1 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_1'
    )(model_in)
    batchnorm_1 = BatchNormalization(name='batchnorm_1')(conv2d_1)
    conv2d_2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_2'
    )(batchnorm_1)
    batchnorm_2 = BatchNormalization(name='batchnorm_2')(conv2d_2)

    maxpool2d_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1')(batchnorm_2)
    dropout_1 = Dropout(0.3, name='dropout_1')(maxpool2d_1)

    conv2d_3 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_3'
    )(dropout_1)
    batchnorm_3 = BatchNormalization(name='batchnorm_3')(conv2d_3)
    conv2d_4 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_4'
    )(batchnorm_3)
    batchnorm_4 = BatchNormalization(name='batchnorm_4')(conv2d_4)

    maxpool2d_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2')(batchnorm_4)
    dropout_2 = Dropout(0.3, name='dropout_2')(maxpool2d_2)

    conv2d_5 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_5'
    )(dropout_2)
    batchnorm_5 = BatchNormalization(name='batchnorm_5')(conv2d_5)
    conv2d_6 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_6'
    )(batchnorm_5)
    batchnorm_6 = BatchNormalization(name='batchnorm_6')(conv2d_6)

    maxpool2d_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3')(batchnorm_6)
    dropout_3 = Dropout(0.3, name='dropout_3')(maxpool2d_3)

    flatten = Flatten(name='flatten')(dropout_3)

    dense_1 = Dense(
        128,
        activation='elu',
        kernel_initializer='he_normal',
        name='dense1'
    )(flatten)
    batchnorm_7 = BatchNormalization(name='batchnorm_7')(dense_1)
    dropout_4 = Dropout(0.4, name='dropout_4')(batchnorm_7)

    model_out = Dense(
        num_classes,
        activation='softmax',
        name='out_layer'
    )(dropout_4)

    model = Model(inputs=model_in, outputs=model_out, name="DCNN")

    return model



if __name__ == '__main__':
   # createEmotionFolder()


   total_images = 0
   for dir_ in os.listdir(INPUT_PATH):
       count = 0
       for f in os.listdir(INPUT_PATH + dir_ + "/"):
           count += 1
       total_images += count
       print(f"{dir_} has {count} number of images")

   print(f"\ntotal images: {total_images}")

   loadImages(total_images)
