# #Unzip the test directory
# !unzip drive/My\ Drive/CatVSDog/test1.zip
# #Unzip the train directory
# !unzip drive/My\ Drive/CatVSDog/train.zip
# Plotting the images of dog
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.models import Sequential
import random
import os
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import save
from numpy import asarray
from os import listdir
from matplotlib import pyplot
from matplotlib.image import imread
folder = 'train/'
for i in range(9):
    # define subplot
    pyplot.subplot(330+1+i)
    # define the filename
    filename = folder + 'dog.'+str(i)+'.jpg'

    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    pyplot.imshow(image)

pyplot.show()
# Plotting the images of cat
folder = 'train/'
for i in range(9):
    # define subplot
    pyplot.subplot(330+1+i)
    # define the filename
    filename = folder + 'cat.'+str(i)+'.jpg'

    # load image pixels
    image = imread(filename)
    # plot raw pixel data
    pyplot.imshow(image)

pyplot.show()
# define location of dataset

folder = 'train/'
photos, labels = list(), list()
# enumerate files in the directory

# for file in listdir(folder):
#   #determine class
#   output = 0.0
#   if file.startswith('cat'):
#     output = 1.0
#   #load image
#   photo = load_img(folder+file,target_size = (200,200))
#   photo = img_to_array(photo)

#   #store
#   photos.append(photo)
#   labels.append(output)

# #convert to a numpy arrays

# photos = asarray(photos)
# labels = asarray(labels)
# print(photos.shape,labels.shape)
# #save the reshaped photos
# save('dogs_vs_cats_photos.npy',photos)
# save('dogs_vs_cats_labels.npy',labels)
# #loading from numpy data
# from numpy import load
# photos = load('dogs_vs_cats_photos.npy')
# labels = load('dogs_vs_cats_labels.npy')

# print(photos.shape,labels.shape)


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

# Partitioning the test and train sets


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


# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(
    200, 200, 3), activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))


# Loading the model
# classifier.load_weights("/kaggle/output/weights.best.hdf5")

# Compiling the CNN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
                                                 target_size=(200, 200),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
                                            target_size=(200, 200),
                                            batch_size=32,
                                            class_mode='binary')

# Select the path to store the final checkpoint after a epoch

filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=50,
                         validation_data=test_set,
                         callbacks=callbacks_list,
                         validation_steps=2000)
