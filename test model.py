# This script will test the model with user uploaded images
# Please save your images in Data/Petimages/external


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


import tensorflow as tf
from keras.preprocessing import image
import numpy as np

path = 'Data\Petimages\external' 

model=tf.keras.models.load_model('Data/trained_model.model')
for img in os.listdir(path):
 
  # predicting images
  img_path=os.path.join(path,img)
  read_img=image.load_img(img_path, target_size=(50, 50))
  
  x=image.img_to_array(read_img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])
  
  classes = model.predict(images, batch_size=10)
  
  print(classes[0])
  
  if classes[0]>0:
    print(img + " is a dog")
    
  else:
    print(img + " is a cat")

