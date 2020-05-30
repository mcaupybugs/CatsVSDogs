from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import shutil
from os import listdir
import random
import os
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')
!unzip drive/My\ Drive/CatVSDog/test1.zip
!unzip drive/My\ Drive/CatVSDog/train.zip
# Alternate method
# creating seperate directory for test->cat and test->dog as this is required
dataset_home = 'dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    labeldirs = ['dogs/', 'cats/']
    for labeldir in labeldirs:
        newdir = dataset_home+subdir+labeldir
        os.makedirs(newdir, exist_ok=True)
print("DONE")


random.seed(1)

val_ratio = 0.25
src_directory = 'train/'
for file in listdir(src_directory):
    src = src_directory+'/'+file
    dst_dir = 'train/'
    if random.random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst = dataset_home+dst_dir+'cats/'+file
        shutil.copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir+'dogs/'+file
        shutil.copyfile(src, dst)

TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
MODEL_FILE = "cats_vs_dogs.h5"

# Data
path = "dataset_dogs_vs_cats/"
training_data_dir = path + "train"  # 10 000 * 2
validation_data_dir = path + "test"  # 2 500 * 2


# Hyperparams
IMAGE_SIZE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 30

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

# CNN Model 5 (https://towardsdatascience.com/image-classifier-cats-vs-dogs-with-convolutional-neural-networks-cnns-and-google-colabs-4e9af21ae7a8)
model = Sequential()

model.add(Conv2D(32, (3, 3), border_mode='same',
                 input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(64, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(128, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), border_mode='same', activation='relu'))
model.add(Conv2D(256, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['accuracy'])

with open(MODEL_SUMMARY_FILE, "w") as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))

    training_data_generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

# Data preparation
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")
validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")

filepath = "/content/drive/My Drive/MyCNN/bestModel.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Training
model.fit_generator(
    training_generator,
    steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1)
model.save_weights(MODEL_FILE)
