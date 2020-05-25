import numpy as np
from keras.preprocessing import image
from keras.models import load_model


classifier = load_model('model.h5')
img_pred = image.load_img('sample_image.jpeg', target_size=(200, 200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)
rslt = classifier.predict(img_pred)

#ind = training_set.class_indices

if rslt[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"

print(prediction)
