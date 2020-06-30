import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

def nvidia_model():
    # Model copied from 
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

    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

    def load_training_data():
        pass

    def preprocess(images):
        pass

    def one_hot(labels):
        pass

if __name__ == '__main__':
    X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_training_data()

    X_train = preprocess(X_train_orig)
    X_test = preprocess(X_test_orig)

    y_train = one_hot(y_train_orig)
    y_test = one_hot(y_test_orig)

    model = nvidia_model()
