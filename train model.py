import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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
        'Data/Petimages/train',  
        target_size=(50, 50),  
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = train_datagen.flow_from_directory(
        'Data/Petimages/test',  
        target_size=(50, 50),  
        class_mode='binary')

history = model.fit(train_generator,validation_data=test_generator,epochs=10,verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.show()
plt.savefig('Training and validation accuracy.png')

model.save('Data/trained_model.model')
