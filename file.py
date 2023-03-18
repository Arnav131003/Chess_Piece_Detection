
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from  keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten,MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from keras.callbacks import EarlyStopping 


import os
for dirname, _, filenames in os.walk('./archive/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



path="./archive/"
from tensorflow.keras.preprocessing.image import ImageDataGenerator




train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)




class_names = ['Queen','Rook','bishop','knight','pawn']
model = tf.keras.models.Sequential()
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(train_data,validation_data = test_data , epochs=21 , verbose =2)


import numpy as np
from tensorflow.keras.preprocessing import image
img_path = 'archive/test_file.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0
plt.imread(img_path)
pred = model.predict(x)
class_idx = np.argmax(pred)
class_labels = ['Queen','Rook','bishop','pawn','knight']
class_label = class_labels[class_idx]
print('Predicted class label:', class_label)



