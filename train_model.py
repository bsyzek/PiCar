import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


TRAINING_DATA_DIR = 'training_data'
PERCENT_TRAIN = 0.99
LABEL_MAP = {'forward': 0, 'right': 1, 'left': 2}
NUM_CATEGORIES = 3

def test_model():
    
    model = Sequential(name='Test_Model')

    model.add(Conv2D(10, 5, padding="same", input_shape=(66, 200, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(20, 5, padding="same", activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(50, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def nvidia_model():
    # Model based on tutorial 
    # https://towardsdatascience.com/deeppicar-part-5-lane-following-via-deep-learning-d93acdce6110
    
    model = Sequential(name='Nvidia_Model')

    # Convolutional layers
    model.add(Conv2D(24, (5, 5), strides=(2,2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2)) # Not in original model
    model.add(Conv2D(64, (3, 3), activation='elu'))

    # Fully connected layers 
    model.add(Flatten())
    model.add(Dropout(0.2)) # Not in original model
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    # Output layer
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def load_training_data():
    images = []
    labels = []
    
    for file in os.listdir(TRAINING_DATA_DIR):
        file_path = TRAINING_DATA_DIR + "\\" + file
        training_data = pickle.load( open(file_path, 'rb'))
        
        for image in training_data["images"]:
                images.append(image)
        
        for label in training_data["labels"]:
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    num_images = labels.shape[0]
    num_training_images = int(num_images * PERCENT_TRAIN)
    training_indices = np.random.choice(num_images, num_training_images, replace=False)

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    
    for idx in range(num_images):
        if idx in training_indices:
            training_images.append(images[idx])
            training_labels.append(labels[idx])
        else:
            testing_images.append(images[idx])
            testing_labels.append(labels[idx])

    return np.array(training_images), np.array(training_labels), np.array(testing_images), np.array(testing_labels)

def preprocess(images):
    processed_images = []
    for image in images: 
        image = image / 255.0
        processed_images.append(image)
    
    return np.array(processed_images)

def one_hot(labels):
    labels_to_ints = []
    for label in labels:
        labels_to_ints.append(LABEL_MAP[label])

    one_hot_labels = tf.one_hot(labels_to_ints, NUM_CATEGORIES)
    return one_hot_labels.numpy()

if __name__ == '__main__':
    X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_training_data()

    X_train = preprocess(X_train_orig)
    X_test = preprocess(X_test_orig)

    y_train = one_hot(y_train_orig)
    y_test = one_hot(y_test_orig)

    model = test_model()
    
    history = model.fit(X_train, y_train, batch_size=100, epochs=10)

    plt.plot(history.history['loss'], color='blue')
    plt.legend(["training loss"])
    plt.show()
    
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()

    # with tf.io.gfile.GFile('nvidia_model_v3.tflite', 'wb') as f:
    #     f.write(tflite_model)