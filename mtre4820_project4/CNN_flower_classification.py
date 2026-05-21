# %%
'''
Members: Lauren Nunez, Regina Martinez, Saam Grami
'''

# %%
'''
1. Generate independent training, validation, and test sets from the dataset.
2. The validation set must contain 500 images, with 100 images from each flower category.
3. The test set must contain 500 images, with 100 images from each flower category.
4. Assign all remaining images to the training set and make sure the three splits are fully independent.
5. Build a CNN with multiple convolution and max-pooling layers.
6. Train on the training set, evaluate on the validation set after each epoch, and report final performance on the test set.
7. Tune the network depth and layer sizes to improve test accuracy.
8. Display the training and validation accuracy histories and print the final test accurac
'''


# %%
'''
Activate venv
    source /home/acrfa/ml-class/.venv/bin/activate
'''

# %%

'''
To get the data set we should install kaggle:
    https://stackoverflow.com/questions/77890757/how-can-i-import-data-from-kaggle-while-not-downloading-it#:~:text=You%20can%20import%20data%20from%20Kaggle%20without,clicking%20**Add%20Data**%20in%20the%20right%20menu
    pip install kaggle #for py
    pip install kagglehub #for notebooks

How to download kaggle libraries into ipynb
    https://www.geeksforgeeks.org/python/how-to-download-kaggle-datasets-into-jupyter-notebook/
    
Different kaggle splitting methods
    https://www.kaggle.com/discussions/general/448072

Kaggle example
    https://www.kaggle.com/code/kanchana1990/exoplanet-intelligence
'''




# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os, shutil
import keras
import random
from pathlib import Path
import cv2


original_dataset_dir = Path('/home/acrfa/ml-class/mtre4820_project4/flowers-dataset/train')
base_dir = Path('/home/acrfa/ml-class/mtre4820_project4/flowers-dataset/flowers_split')


classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# slashes, '/', join paths
train_dir = base_dir / 'train'
val_dir = base_dir / 'validation'
test_dir = base_dir / 'test'

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

#Creates all folders for splits

#defines the daisy validation path
val_daisy_dir = base_dir / 'validation' / classes[0]
#defines the daisy test path
test_daisy_dir = base_dir / 'test' / classes[0]
#defines the daisy train path
train_daisy_dir = base_dir / 'train' / classes[0]


#makes the daisy validation directory
val_daisy_dir.mkdir(parents=True, exist_ok=True ) #Essentially this says create a folder if nothing there or leave the folder if it is there
#makes the daisy test directory
test_daisy_dir.mkdir(parents=True, exist_ok=True)
#makes the daisy train directory
train_daisy_dir.mkdir(parents=True, exist_ok=True)



#defines the daisy validation path
val_dandelion_dir = base_dir / 'validation' / classes[1]
#defines the daisy test path
test_dandelion_dir = base_dir / 'test' / classes[1]
#defines the daisy train path
train_dandelion_dir = base_dir / 'train' / classes[1]


#makes the daisy validation directory
val_dandelion_dir.mkdir(parents=True, exist_ok=True ) #Essentially this says create a folder if nothing there or leave the folder if it is there
#makes the daisy test directory
test_dandelion_dir.mkdir(parents=True, exist_ok=True)
#makes the daisy train directory
train_dandelion_dir.mkdir(parents=True, exist_ok=True)




#defines the daisy validation path
val_rose_dir = base_dir / 'validation' / classes[2]
#defines the daisy test path
test_rose_dir = base_dir / 'test' / classes[2]
#defines the daisy train path
train_rose_dir = base_dir / 'train' / classes[2]


#makes the daisy validation directory
val_rose_dir.mkdir(parents=True, exist_ok=True ) #Essentially this says create a folder if nothing there or leave the folder if it is there
#makes the daisy test directory
test_rose_dir.mkdir(parents=True, exist_ok=True)
#makes the daisy train directory
train_rose_dir.mkdir(parents=True, exist_ok=True)



#defines the daisy validation path
val_sunflower_dir = base_dir / 'validation' / classes[3]
#defines the daisy test path
test_sunflower_dir = base_dir / 'test' / classes[3]
#defines the daisy train path
train_sunflower_dir = base_dir / 'train' / classes[3]


#makes the daisy validation directory
val_sunflower_dir.mkdir(parents=True, exist_ok=True ) #Essentially this says create a folder if nothing there or leave the folder if it is there
#makes the daisy test directory
test_sunflower_dir.mkdir(parents=True, exist_ok=True)
#makes the daisy train directory
train_sunflower_dir.mkdir(parents=True, exist_ok=True)



#defines the daisy validation path
val_tulip_dir = base_dir / 'validation' / classes[4]
#defines the daisy test path
test_tulip_dir = base_dir / 'test' / classes[4]
#defines the daisy train path
train_tulip_dir = base_dir / 'train' / classes[4]


#makes the daisy validation directory
val_tulip_dir.mkdir(parents=True, exist_ok=True ) #Essentially this says create a folder if nothing there or leave the folder if it is there
#makes the daisy test directory
test_tulip_dir.mkdir(parents=True, exist_ok=True)
#makes the daisy train directory
train_tulip_dir.mkdir(parents=True, exist_ok=True)

for cls in classes:
    cls_train_dir = train_dir / cls
    cls_val_dir = val_dir / cls
    cls_test_dir = test_dir / cls

    cls_train_dir.mkdir(parents=True, exist_ok=True)
    cls_val_dir.mkdir(parents=True, exist_ok=True)
    cls_test_dir.mkdir(parents=True, exist_ok=True)

    print("Created:", cls_train_dir, cls_val_dir, cls_test_dir)

size_of_flowers = [0,0,0,0,0]

for i in range(len(classes)):
    counter = 0

    #goes into dasiy file and stores every pic as a list
    pics = list((original_dataset_dir / classes[i]).iterdir())
    #shuffle so not same pic everytime
    random.shuffle(pics)

    #loop through first 100 pics for validation set (will do 100 from each type, 500 total) and puts in proper folder
    for img in pics[:100]:
        (base_dir / 'validation' / classes[i] / img.name).write_bytes(img.read_bytes())
        counter+=1
    #loops through next 100 pics for test set (will do 100 from each type, 500 total) and puts in proper folder
    for img in pics[100:200]:
        (base_dir / 'test' / classes[i] / img.name).write_bytes(img.read_bytes())
        counter+=1
    #assign all remaining images to training set
    for img in pics[200:]:
        (base_dir / 'train' / classes[i] / img.name).write_bytes(img.read_bytes())
        counter+=1
    size_of_flowers[i] = counter

print(size_of_flowers)


# %%
#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import os, shutil
import keras
import random
from pathlib import Path
import cv2

# %%

#Create folders for each class 
original_dataset_dir = Path('/home/acrfa/ml-class/mtre4820_project4/flowers-dataset/train')
base_dir = Path('/home/acrfa/ml-class/mtre4820_project4/flowers-dataset/flowers_split4')
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Create top-level split directories
for split in ['train', 'validation', 'test']:
    (base_dir / split).mkdir(parents=True, exist_ok=True)

# Create a subdirectory for each class within each split
for cls in classes:
    for split in ['train', 'validation', 'test']:
        (base_dir / split / cls).mkdir(parents=True, exist_ok=True)



for set in ['train', 'validation', 'test']:             # Splits data into three folders
    (base_dir / set).mkdir(parents=True, exist_ok=True)

for flower in classes:                                  # Puts flower sub folder into each set
    for flower in ['train', 'validation', 'test']:
        (base_dir / split / cls).mkdir(parents=True, exist_ok=True)

for cls in classes:
    cls_train_dir = train_dir / cls
    cls_val_dir = val_dir / cls
    cls_test_dir = test_dir / cls

    cls_train_dir.mkdir(parents=True, exist_ok=True)
    cls_val_dir.mkdir(parents=True, exist_ok=True)
    cls_test_dir.mkdir(parents=True, exist_ok=True)

    print("Created:", cls_train_dir, cls_val_dir, cls_test_dir)



# %%
#Randomizes and sorts pictures into training, validation, and test
size_of_flowers = [0,0,0,0,0]

for i in range(len(classes)):
    counter = 0

    #goes into dasiy file and stores every pic as a list
    pics = list((original_dataset_dir / classes[i]).iterdir())
    #shuffle so not same pic everytime
    random.shuffle(pics)

    #loop through first 100 pics for validation set (will do 100 from each type, 500 total) and puts in proper folder
    for img in pics[:100]:
        (base_dir / 'validation' / classes[i] / img.name).write_bytes(img.read_bytes())
        counter+=1
    #loops through next 100 pics for test set (will do 100 from each type, 500 total) and puts in proper folder
    for img in pics[100:200]:
        (base_dir / 'test' / classes[i] / img.name).write_bytes(img.read_bytes())
        counter+=1
    #assign all remaining images to training set
    for img in pics[200:]:
        (base_dir / 'train' / classes[i] / img.name).write_bytes(img.read_bytes())
        counter+=1
    size_of_flowers[i] = counter

print(size_of_flowers)

# %%
#create batches for the cnn

train_ds = image_dataset_from_directory(
    "/home/acrfa/ml-class/mtre4820_project4/flowers-dataset/flowers_split/train/",
    image_size=(224,224),
    batch_size=32
)

validation_ds = image_dataset_from_directory(
    "/home/acrfa/ml-class/mtre4820_project4/flowers-dataset/flowers_split/validation/",
    image_size=(224,224),
    batch_size=32
)

test_ds = image_dataset_from_directory(
    "/home/acrfa/ml-class/mtre4820_project4/flowers-dataset/flowers_split/test/",
    image_size=(224,224),
    batch_size=32
)




# %%
#Check batch amounts and labels
for i, (images, labels) in enumerate(train_ds):
    print(f"Batch {i+1}")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels.numpy())

for i, (images, labels) in enumerate(validation_ds):
    print(f"Batch {i+1}")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels.numpy())
for i, (images, labels) in enumerate(test_ds):
    print(f"Batch {i+1}")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels.numpy())



# %%
#CNN
#https://www.geeksforgeeks.org/deep-learning/convolutional-neural-network-cnn-in-tensorflow/
import tensorflow as tf
import matplotlib.pyplot as plt
 

#CNN definition
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255,   input_shape=(224,224,3)),                   # Normalizes our images to 0 t01
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)), # 64 filters and relu goes over the dot products
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),                                 # Picks the largest number out of 4
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),                                                       # Preps it for the neural network
    tf.keras.layers.Dense(128, activation='relu'),                                   # Goes to 128 hidden layer
    tf.keras.layers.Dense(5, activation='softmax')                                   # 5 outputs, 1 per class and has a softmax
])

model.compile(     #This is how we do our backprop
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # because labels are integers
    metrics=['accuracy']
)

# Assuming you already have train_ds, validation_ds, test_ds
history = model.fit(
    train_ds,                  # contains both images and labels
    validation_data=validation_ds,
    epochs=10
)


# Optional: Plot training/validation accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_ds)
print("Test accuracy:", test_accuracy)


