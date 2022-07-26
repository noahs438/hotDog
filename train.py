import gc
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K

# Image paths
trainDataDir = 'train'
valDataDir = 'validation'
numberTrainSample = 3995
numberValidationSample = 497
epochs = 50  # Diminishing returns when increasing epochs too much
batchSize = 5

# Set the size of the image used throughout the process
imgWidth, imgHeight = 150, 150

verbose = 1


def build_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, imgWidth, imgHeight)
    else:
        input_shape = (imgWidth, imgHeight, 3)

    # Create initial model
    model = Sequential()

    # First conv. layer
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second conv. layer
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third conv. layer
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten + implement Dropout (eliminate 50% of neurons)
    # Output layer has a single neuron which determines whether we have a hotDog or notHotDog
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compiling the model using binary cross entropy as the loss function and the rmsprop optimizer
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def train_model(model):
    # Augmentation configuration used for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Augmentation configuration used for testing - ONLY rescaling
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_generator = train_datagen.flow_from_directory(
        trainDataDir,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        valDataDir,
        target_size=(imgWidth, imgHeight),
        batch_size=batchSize,
        class_mode='binary'
    )

    model.fit(
        train_generator,
        steps_per_epoch=numberValidationSample // batchSize,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=numberValidationSample // batchSize
    )

    return model


# Machine learning model filename
def save_model(model):
    model.save('saved_model.h5')


def main():
    myModel = None
    K.clear_session()
    gc.collect()
    myModel = build_model()
    myModel = train_model(myModel)
    save_model(myModel)


main()