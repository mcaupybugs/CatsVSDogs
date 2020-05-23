# #Unzip the test directory
# !unzip drive/My\ Drive/CatVSDog/test1.zip
# #Unzip the train directory
# !unzip drive/My\ Drive/CatVSDog/train.zip
#Plotting the images of dog
from matplotlib import pyplot
from matplotlib.image import imread
folder = 'train/'
for i in range(9):
  #define subplot 
  pyplot.subplot(330+1+i)
  #define the filename 
  filename =  folder + 'dog.'+str(i)+'.jpg'

  #load image pixels
  image = imread(filename)
  #plot raw pixel data
  pyplot.imshow(image)

pyplot.show()
#Plotting the images of cat
from matplotlib import pyplot
from matplotlib.image import imread
folder = 'train/'
for i in range(9):
  #define subplot 
  pyplot.subplot(330+1+i)
  #define the filename 
  filename =  folder + 'cat.'+str(i)+'.jpg'

  #load image pixels
  image = imread(filename)
  #plot raw pixel data
  pyplot.imshow(image)

pyplot.show()
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#define location of dataset

folder = 'train/'
photos,labels = list(),list()
#enumerate files in the directory

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


import os
#Alternate method
#creating seperate directory for test->cat and test->dog as this is required
dataset_home='dataset_dogs_vs_cats/'
subdirs = ['train/','test/']
for subdir in subdirs:
    labeldirs = ['dogs/','cats/']
    for labeldir in labeldirs:
        newdir = dataset_home+subdir+labeldir
        os.makedirs(newdir,exist_ok=True)
print("DONE")

#Partitioning the test and train sets 
import random
import shutil 

from os import listdir

random.seed(1)

val_ratio = 0.25
src_directory = 'train/'
for file in listdir(src_directory):
    src=src_directory+'/'+file
    dst_dir = 'train/'
    if random.random() < val_ratio:
        dst_dir='test/'
    if file.startswith('cat'):
        dst = dataset_home+dst_dir+'cats/'+file
        shutil.copyfile(src,dst)
    elif file.startswith('dog'):
        dst = dataset_home +dst_dir+'dogs/'+file
        shutil.copyfile(src,dst)


#CNN model 
from tensorflow import keras
from tensorflow.keras import layers
def define_model():
  model = keras.Sequential()
  model.add(layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same',input_shape=(200,200,3)))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(128,activation='relu',kernel_initializer='he_uniform'))
  model.add(layers.Dense(1,activation='sigmoid'))

  #Compile model
  opt = keras.optimizers.SGD(lr=0.001,momentum=0.9)
  model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
  return model

model = define_model()

#Converting the image pixels to scale between 0 and 1
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1.0/255.0)

#making test and train sets
train_it  = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
                                        class_mode='binary',batch_size=64,target_size=(200,200))
test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
                                      class_mode='binary',batch_size=64,target_size=(200,200))

history = model.fit_generator(train_it,steps_per_epoch=len(train_it),
                              validation_data=test_it,validation_steps=len(test_it),epochs=20,verbose=0)
# evaluate model
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))