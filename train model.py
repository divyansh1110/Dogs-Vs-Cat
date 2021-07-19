import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model=tf.keras.models.Sequential([
                                tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(50,50,3)),
                                tf.keras.layers.MaxPool2D(2,2),

                                tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                tf.keras.layers.MaxPool2D(2,2), 
                                
                                tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
                                tf.keras.layers.MaxPool2D(2,2), 

                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(128,activation='relu'),
                                tf.keras.layers.Dense(1,activation='sigmoid'),
                                
                                

])
print(model.summary())
model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')


train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'Data/Petimages/',  # This is the source directory for training images
        target_size=(50, 50),  # All images will be resized to 100x100
        # batch_size=64,
        class_mode='binary')


history = model.fit(train_generator,epochs=25,verbose=1)