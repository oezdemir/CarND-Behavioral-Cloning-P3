import csv
import cv2
import matplotlib.image as mpimg
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Cropping2D, Lambda, Conv2D, ELU
from keras.optimizers import Adam
from keras import applications
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# GLOBAL VARS
CSV_PATH = 'data/driving_log.csv'
IMG_PATH = 'data/IMG/'
BATCH_SIZE = 40

def get_samples(csv_path, IMG_PATH):
    samples = []
    with open(CSV_PATH) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            line['center'] = IMG_PATH + line['center'].split('/')[-1]
            line['left'] = IMG_PATH + line['left'].split('/')[-1]
            line['right'] = IMG_PATH + line['right'].split('/')[-1]
            samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples


def augment_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def augment_translation(image, angle):
    delta_x = 100
    delta_y = 10
    trans_x = delta_x * (np.random.rand() - 0.5)
    trans_y = delta_y * (np.random.rand() - 0.5)
    angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, angle

def augment_flip(image, angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle

def augment(sample):
    # Most of the driving data has a steering wheel angle of 0.
    # To keep the training data balanced some of the training images will not be augmented.
    if np.random.rand() < 0.55:
        image = mpimg.imread(sample['center'])
        angle = float(sample['steering'])
        return image, angle

    choices = ['left', 'right', 'center']
    choice = choices[np.random.choice(3)]
    image = mpimg.imread(sample[choice])
    angle = float(sample['steering'])
    # Adjust steering angle based on camera position as part of augmentation
    if choice == 'left':
        angle += 0.20
    elif choice == 'right':
        angle -= 0.20

    image, angle = augment_flip(image, angle)
    image, angle = augment_translation(image, angle)
    image = augment_brightness(image)
    return image, angle


def generate(samples, augmented, batch_size=32):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image, angle = (None, None)
                if augmented:
                    image, angle = augment(batch_sample)
                else:
                    image = mpimg.imread(batch_sample['center'])
                    angle = float(batch_sample['steering'])
                
                images.append(image)
                angles.append(angle)
            X = np.array(images)
            y = np.array(angles)
            yield X, y

## MODEL

def resize(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.resize_images(image, [66, 200])

def to_gray(image):
    import tensorflow as tf  # This import is required here otherwise the model cannot be loaded in drive.py
    return tf.image.rgb_to_grayscale(image)

def preprocess_model(model):
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3), name='crop')) 
    model.add(Lambda(resize, name='resize'))
#    model.add(Lambda(to_gray, name='grayscale'))
    model.add(Lambda(lambda x: (x/255.0) - 0.5, name='normalize'))
    return model

def nvidia_model():
    model = Sequential()
    model = preprocess_model(model)
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model



## TRAINING
from keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, train_samples, validation_samples):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    history = model.fit_generator( generator=generate(train_samples, True, BATCH_SIZE),
                     steps_per_epoch=len(train_samples) // (BATCH_SIZE), 
                     validation_data=generate(validation_samples, False, BATCH_SIZE),
                     validation_steps=len(validation_samples) // (BATCH_SIZE),
                     epochs=10,
                     verbose=1,
                     callbacks = [checkpoint] # [early_stopping, checkpoint]
                    )

    model.save('model.h5')
    print('Model saved.')
    return history


def main():
    # Load data
    train_samples, validation_samples = get_samples(CSV_PATH, IMG_PATH)
    model = nvidia_model()
    model.compile(optimizer='adam', loss='mse')

    # Train
    history = train_model(model, train_samples, validation_samples)

    # Visualize
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import random

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()


if __name__ == '__main__':
    main()

