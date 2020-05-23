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

for file in listdir(folder):
  #determine class 
  output = 0.0
  if file.startswith('cat'):
    output = 1.0
  #load image
  photo = load_img(folder+file,target_size = (200,200))
  photo = img_to_array(photo)

  #store
  photos.append(photo)
  labels.append(output)

#convert to a numpy arrays

photos = asarray(photos)
labels = asarray(labels)
print(photos.shape,labels.shape)
#save the reshaped photos
save('dogs_vs_cats_photos.npy',photos)
save('dogs_vs_cats_labels.npy',labels)
