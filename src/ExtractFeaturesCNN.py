'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import LSTM
from keras import applications
from keras.models import model_from_json
import os
import cv2
from sklearn.metrics import confusion_matrix
# dimensions of our images.
img_width, img_height = 100,100

top_model_weights_path = '../bottleneck_fc_modelCONFCohen1.h5'
train_data_dir = '../dataFromPython3/train'
validation_data_dir = '../dataFromPython3/validation'

test_data_dir = '../dataFromPython3/validation'
nb_train_samples =144#2045 #144 #2045 #2290# 168# 2290 #8 #2318
nb_validation_samples = 35 #689 #689 #35 #689 #770 #41# 770 #1
epochs = 100
batch_size = 1#10

def save_bottlebeck_featuresTest():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, 38 // batch_size)
    np.save(open('bottleneck_features_testCONFTESTJAFFE1.npy', 'wb'),
            bottleneck_features_train)



def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('../bottleneck_features_trainCONFJAFFE1.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, 38 // batch_size)
    np.save(open('../bottleneck_features_validationCONFJAFFE1.npy', 'wb'),
            bottleneck_features_validation)



def train_top_model():
    path = 'dataFromPython3/train/'
    # classify each image in the validation set
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
    emotionLbls = [[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0],[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]] #,[0,0,0,0,0,0,1]]
    train_data = np.load(open('bottleneck_features_trainCONFJAFFE1.npy', 'rb'))
    labelbyemotion = []
    for emotion in emotions:
        for file in os.listdir(path+emotion):
            if 'png' in file:
                labelbyemotion.append(emotionLbls[emotions.index(emotion)])
    train_labels = np.array(labelbyemotion)

    #[0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validationCONFJAFFE1.npy', 'rb'))
    #validation_labels = np.array(
    #    [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    path = 'dataFromPython3/validation/'
    labelbyemotionVal = []
    for emotion in emotions:
        for file in os.listdir(path + emotion):
            if 'png' in file:
                labelbyemotionVal.append(emotionLbls[emotions.index(emotion)])
    validation_labels = np.array(labelbyemotionVal)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(1024, activation='relu'))

    #model.add(Dense(1024, activation='relu'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

    # serialize model to JSON

    model_json = model.to_json()
    with open( "PretrainCNNCONFJAFFE1.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("PreTraineCNNWeightCONFJAFFE1.h5")
    print("Saved model to disk")


#save_bottlebeck_features()
#train_top_model()


if __name__ == '__main__':
    # save_bottlebeck_featuresTest()
#     save_bottlebeck_features()
    # print("features saved")
    path = '../dataFromPython3/train/'
    # classify each image in the validation set
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
    emotionLbls = [[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0],[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]] #,[0,0,0,0,0,0,1]]
    train_data = np.load(open('../bottleneck_features_trainCONFJAFFE1.npy', 'rb'))
    labelbyemotion = []
    for emotion in emotions:
        for file in os.listdir(path+emotion):
            if 'png' in file:
                labelbyemotion.append(emotionLbls[emotions.index(emotion)])
    train_labels = np.array(labelbyemotion)
 
    #[0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
 
    validation_data = np.load(open('../bottleneck_features_validationCONFJAFFE1.npy', 'rb'))
    #validation_labels = np.array(
    #    [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    path = '../dataFromPython3/validation/'
    labelbyemotionVal = []
    for emotion in emotions:
        for file in os.listdir(path + emotion):
            if 'png' in file:
                labelbyemotionVal.append(emotionLbls[emotions.index(emotion)])
    validation_labels = np.array(labelbyemotionVal)
 
    X_train = np.zeros((144, 3, 3, 512), dtype=np.uint8)
    X_train = np.reshape(train_data, (144, 3, 3*512))
    X_test = np.zeros((38, 3, 3, 512), dtype=np.uint8)
    X_test = np.reshape(validation_data, (38, 3, 3*512))
    model = Sequential()
    # model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(LSTM(2048, return_sequences=False, input_shape=X_train.shape[1:],
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
 
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
 
    model.fit(X_train, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_test, validation_labels))
    score = model.evaluate(X_test, validation_labels)
    print(score)
    model.save_weights(top_model_weights_path)

    # serialize model to JSON
    model_json = model.to_json()
    with open( "../PretrainLSMTCONFJAFFE1.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../PreTraineLSMTWeightCONFJAFFE1.h5")
    print("Saved model to disk")