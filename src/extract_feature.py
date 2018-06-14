import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications

# dimensions of our images.
img_width, img_height = 100,100
train_data_dir = '../dataFromPython3/train'
validation_data_dir = '../dataFromPython3/validation'
test_data_dir = '../dataFromPython3/validation'
nb_train_samples =144#2045 #144 #2045 #2290# 168# 2290 #8 #2318
nb_validation_samples = 38 #689 #689 #35 #689 #770 #41# 770 #1
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
        generator, nb_validation_samples // batch_size)
    np.save(open('../bottleneck_features_test.npy', 'wb'),
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
    np.save(open('../bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('../bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

if __name__ == '__main__':
    save_bottlebeck_featuresTest()
    save_bottlebeck_features()
#     print("features saved")