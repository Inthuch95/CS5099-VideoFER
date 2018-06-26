import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.applications.vgg16 import VGG16, preprocess_input
import os

# dimensions of our images.
img_width, img_height = 100,100
train_data_dir = '../dataset/Train/'
validation_data_dir = '../dataset/Validation/'
test_data_dir = '../dataset/Test/'
nb_train_samples = 6528
nb_validation_samples = 364
nb_test_samples = 365
batch_size = 1

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
        generator, nb_test_samples // batch_size)
    np.save(open('../dataset/bottleneck_features_test.npy', 'wb'),
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
    np.save(open('../dataset/bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('../dataset/bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    
def save_labels_onehot():
    # labels with onehot encoding (neural network)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    emotionLbls = [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0], [0,0,1,0,0,0,0],[0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]
    # save training labels
    y_train = create_label_onehot(emotions, emotionLbls, train_data_dir)
    np.save(open('../dataset/y_train.npy', 'wb'), y_train)
    
    # save validation labels
    y_val = create_label_onehot(emotions, emotionLbls, validation_data_dir)
    np.save(open('../dataset/y_val.npy', 'wb'), y_val)
    
    # save validation labels
    y_test = create_label_onehot(emotions, emotionLbls, test_data_dir)
    np.save(open('../dataset/y_test.npy', 'wb'), y_test)
    
def save_labels_text():
    # Create number labels (SVM)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    # save training labels
    y_train = create_label_text(emotions, train_data_dir)
    np.save(open('../dataset/y_train_text.npy', 'wb'), y_train)
    
    # save validation labels
    y_val = create_label_text(emotions, validation_data_dir)
    np.save(open('../dataset/y_val_text.npy', 'wb'), y_val)
    
    # save validation labels
    y_test = create_label_text(emotions, test_data_dir)
    np.save(open('../dataset/y_test_text.npy', 'wb'), y_test)
    
def create_label_onehot(emotions, emotionLbls, data_dir):
    labelbyemotion = []
    for emotion in emotions:
        for file in os.listdir(data_dir+emotion):
            if 'jpg' in file:
                labelbyemotion.append(emotionLbls[emotions.index(emotion)])
    labels = np.array(labelbyemotion)
    return labels

def create_label_text(emotions, data_dir):
    labelbyemotion = []
    for emotion in emotions:
        for file in os.listdir(data_dir+emotion):
            if 'jpg' in file:
                labelbyemotion.append(emotions[emotions.index(emotion)])
    labels = np.array(labelbyemotion)
    return labels

if __name__ == '__main__':
    save_bottlebeck_features()
    save_bottlebeck_featuresTest()
    save_labels_onehot()
    save_labels_text()