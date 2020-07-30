"""<=============================================== IMPORTING LIBRARIES ===============================================>"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""<=============================================== IMPORTING AND PRE-PROCESSING DATA ===============================================>"""

_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_dir = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=_URL, extract=True)

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats') # cat training pictures
train_dogs_dir = os.path.join(train_dir, 'dogs') # dog training pictures
validation_cats_dir = os.path.join(validation_dir, 'cats') # cat validation pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs') # dog validation pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

"""<=============================================== RESIZING IMAGES ===============================================>"""

BATCH_SIZE = 100 #Training examples
IMG_SHAPE = 150 #length and width of image

train_image_generator = ImageDataGenerator(rescale=1./255) # Training Data Generator
validation_image_generator = ImageDataGenerator(rescale=1./255) # Validation Data Generator

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True, 
                                                           target_size=(IMG_SHAPE, IMG_SHAPE), 
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=validation_dir,
                                                           shuffle=False, 
                                                           target_size=(IMG_SHAPE, IMG_SHAPE), 
                                                           class_mode='binary')

sample_training_images, _ = next(train_data_gen)

"""<=============================================== BUILDING THE NEURAL NETWORK ===============================================>"""

cats_dogs_model = tf.keras.Sequential([
                                       tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
                                       tf.keras.layers.MaxPooling2D(2, 2), 

                                       tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                       tf.keras.layers.MaxPooling2D(2, 2), 

                                       tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                                       tf.keras.layers.MaxPooling2D(2, 2), 

                                       tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                                       tf.keras.layers.MaxPooling2D(2, 2), 

                                       tf.keras.layers.Flatten(),
                                       tf.keras.layers.Dense(512, activation='relu'),
                                       tf.keras.layers.Dense(2, activation='softmax')

])

cats_dogs_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

"""<=============================================== TRAINING THE MODEL ===============================================>"""

EPOCHS = 100
history = cats_dogs_model.fit_generator(
    train_data_gen,
    steps_per_epoch = int(np.ceil(total_train/float(BATCH_SIZE))),
    epochs = EPOCHS,
    validation_data = val_data_gen,
    validation_steps = int(np.ceil(total_val/float(BATCH_SIZE)))
)

"""<=============================================== RESULTS ===============================================>"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()
