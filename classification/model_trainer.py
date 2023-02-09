from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPooling2D, Flatten, Dropout, Activation
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import SGD

def fit_and_save(epochs,name,epoch_steps,validation_steps):
    history = model.fit_generator(
        train_images,
        epochs=epochs,
        validation_data=test_images,
        steps_per_epoch=epoch_steps,
        validation_steps=validation_steps)
    model.save(name)
    return history

def generate_model(hidden_activation,output_activation):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(24, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation=output_activation))
    opt = SGD(lr=0.01, momentum=0.4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generate_datagen():
    return ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
def image_data_gen():
    return ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


model = generate_model('relu','softmax')
datagen = generate_datagen()

batch_size = 64

train_datagen = image_data_gen()
test_datagen = image_data_gen()

path = 'C:\\Users\\Alonso\\PycharmProjects\\perceptionML'
train_images = train_datagen.flow_from_directory(
    path + "\\newImages",
    target_size=(256,256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_images = test_datagen.flow_from_directory(
    path + "\\images_test",
    target_size=(256,256),
    batch_size=batch_size,
    class_mode='categorical'
)

histories = []
with tf.device("gpu:0"):
    histories.append(fit_and_save(40,'./models_1/model_1.h5',len(train_images),len(test_images)))
    histories.append(fit_and_save(50,'./models_1/model_2.h5',len(train_images),len(test_images)))
    histories.append(fit_and_save(60,'./models_1/model_3.h5',len(train_images),len(test_images)))
    histories.append(fit_and_save(70,'./models_1/model_4.h5',len(train_images),len(test_images)))
    histories.append(fit_and_save(80,'./models_1/model_5.h5',len(train_images),len(test_images)))

i = 0
for history in histories:
    pd.DataFrame(history.history).plot()
    plt.savefig('./models_1/history_' + str(i))
    i+=1