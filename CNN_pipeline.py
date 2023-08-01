import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import imghdr

data_dir_train = r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\archive\DATASET\TRAIN'
data_dir_test = r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation\archive\DATASET\TEST'
'''
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)

        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)

            if tip not in image_exts:
                print("Image extension not approved: " + image_path)
                os.remove(image_path)

        except Exception as e:
            print("Issue with image: " + image_path)
'''
import numpy as np
from matplotlib import pyplot as plt

#Automatically reshapes images to consistent size, consistent number per batch
data = tf.keras.utils.image_dataset_from_directory(data_dir_train, shuffle=True)
test_data = tf.keras.utils.image_dataset_from_directory(data_dir_test, shuffle=True)

#Shuffling data beforehand to prevent bias - IMPORTANT
#data = tf.data.Dataset.shuffle(data, buffer_size=22564)

data_iterator = data.as_numpy_iterator()
test_data_iterator = test_data.as_numpy_iterator()

#Gets new batches from iterator
batch = data_iterator.next()
test_batch = test_data_iterator.next()

#print(batch)
#print(len(batch))
#print(batch[0].shape)   #4 dimensions --> (no. of images per batch, batch dimension 1, batch dimension 2, RGB channels)
#print(batch[1])
#print(batch[0].min())
#print(batch[0])
#Enumeration:   Organic - 0, Recyclable - 1

'''
For Visualisation Purposes:-

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()
'''

#Scaling data so that it is between 0 and 1
scaled_data = data.map(lambda x, y: (x/255, y))
scaled_test_data = test_data.map(lambda x, y: (x/255, y))

#print(data.as_numpy_iterator().next()[0].max())

scaled_iterator = scaled_data.as_numpy_iterator()
scaled_test_iterator = scaled_test_data.as_numpy_iterator()

scaled_data_batch = scaled_iterator.next()
scaled_test_batch = scaled_test_iterator.next()

#Splitting data into training and validation (90:10)
train_size = int(len(data) * 0.9)
val_size = int(len(data) * 0.1)

#print(train_size, val_size)

#Allocating data based on split
train_data = data.take(train_size)
val_data = data.skip(train_size).take(val_size)

#Importing CNN model creation libraries
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

#Creating CNN model
model = Sequential()

#Customizing CNN model (3 convolution layers, 1 flatten layer, 2 dense layers) 
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compiling all layers
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

#Training the model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train_data, epochs=22, validation_data=val_data, callbacks=[tensorboard_callback])

print(hist.history)

from keras.models import load_model

#Model Serialization (h5 file) - similar to .zip
model.save(os.path.join(r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation', r'waste_segregation_model_v3.h5'))

#Plotting performance
#Loss metrics
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#Accuracy metrics
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#Evaluation
from keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, roc_curve, classification_report

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in scaled_test_data.as_numpy_iterator():
    X, y = batch
    yhat= model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

#All 3 values are between 0 and 1, higher the metrics, better the model performance
print(f'Model Precision: {pre.result().numpy()},Model Recall: {re.result().numpy()},Model Accuracy: {acc.result().numpy()}')
'''
pred = model.predict(scaled_test_data)
label = scaled_test_data.label
acc_score = accuracy_score(label, pred)

prec_score = precision_score(y_test, pred)

rec_score = recall_score(y_test, pred)
print("Accuracy score of model: " , acc_score)
print("Precision score of model: " , prec_score)
print("Recall score of model: " , rec_score)
'''

for images, labels in scaled_test_data.map(lambda x, y: (x, y)):
    test_loss, test_acc = model.evaluate(images, labels, verbose=2)      
'''MAKE MANUAL TEST USING UNKNOWN IMAGES OF BOTH CLASSES'''

#Saving the model - Able to be used at a future date
from keras.models import load_model

#Model Serialization (h5 file) - similar to .zip
model.save(os.path.join(r'C:\Users\prith\OneDrive\Desktop\PRITHVIRAJ MODY\Computer Codes\Waste_Segregation', r'waste_segregation_model.h5'))

#Now the model can be instatiated to an object and can predict using .predict()