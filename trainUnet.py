# -*- coding: utf-8 -*-

# Image segmentation with a U-Net-like architecture
# Train the model 

"""## Prepare paths of input images and target segmentation masks"""
from time import time
from unet import iou
from loss_functions import dice_loss, surface_loss_keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

tf.random.set_seed(1337)
AUTOTUNE = tf.data.AUTOTUNE

import os
#uncomment to force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

input_dir = pathlib.Path("/home/dan/dem_site_project/datacrop_3band_minmaxscaler/")
target_dir =  pathlib.Path("/home/dan/dem_site_project/labelcrop_filled/")

BACKBONE = 'resnet34'
label = "resnet34_dicesloss_512crop_filled"
img_size = (512, 512)
#the background class and the particle class
num_classes = 3 
batch_size = 10

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

# input_img_paths = tf.data.Dataset.list_files(str(input_dir+'*.png'), shuffle=False)
# target_img_paths= tf.data.Dataset.list_files(str(target_dir+'*.png'), shuffle=False)

dataset = tf.data.Dataset.from_tensor_slices((input_img_paths,target_img_paths))
for input_path, target_path in dataset.take(4):
    print(input_path, "|", target_path)

length = tf.data.experimental.cardinality(dataset).numpy()

val_size = int(length * 0.1)
train_ds = dataset.skip(val_size)
val_ds = dataset.take(val_size)

print("train size",tf.data.experimental.cardinality(train_ds).numpy())
print("validation size",tf.data.experimental.cardinality(val_ds).numpy())

def decode_imgs(img,mask):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.read_file(img)
  img = tf.io.decode_png(img,channels=3)
  img = img/255
  mask = tf.io.read_file(mask)
  mask = tf.io.decode_png(mask,channels=3)

  # Resize the image to the desired size
  return tf.image.resize(img, img_size), tf.image.resize(mask, img_size)

for input_path, target_path in train_ds.take(10):
    print(input_path, "|", target_path)

train_ds = train_ds.map(decode_imgs, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(decode_imgs, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.shuffle(length,seed=1337)
val_ds = val_ds.shuffle(length,seed=1337)

train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
  
# train_ds = train_ds.repeat() #repeat forever
# val_ds = val_ds.repeat() #repeat forever


for image, mask in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("mask shape: ", mask.numpy().shape)


#print a partial list of found images
print("Number of samples:", len(input_img_paths), len(target_img_paths))


# from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

from unet import dataGenerator
from tensorflow.keras import layers

from unet import get_model
import matplotlib.pyplot as plt


# Build model
# model = get_model(img_size, num_classes)
# model.summary()

"""## Set aside a validation split"""

import random

for image,mask in train_ds.take(1):
  print(image.shape)
  print(np.max(image))
# plt.imshow(w[1][0])
# plt.show()

"""## Train the model"""



# from keras_loss import surface_loss_keras
# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
import segmentation_models as sm
# from unet import get_model
model = sm.Unet(BACKBONE, classes=num_classes, input_shape=img_size+(3,), encoder_weights=None)
# model = get_model(img_size,3)
# model.summary()
print("exit callbacks")
# model.compile(optimizer="adam", loss=sm.losses.DiceLoss(), metrics=[sm.metrics.IOUScore(), tf.keras.metrics.IoU(num_classes=3, target_class_ids=[0,1])])
model.compile(optimizer="adam", loss=sm.losses.DiceLoss(), metrics=[sm.metrics.IOUScore(), tf.keras.metrics.Accuracy()])


def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask


rows = 4
columns = 3
fig_disp,axes_disp = plt.subplots(rows,columns)
def show_predictions(dataset=None, num=1,block=False):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      fig = plt.figure(figsize=(10, 7))

      # display([image[0], mask[0], create_mask(pred_mask)])
      # showing image
      plt.imshow(image[0])  
      plt.axis('off') 
      plt.title("image")  

      fig.add_subplot(rows, columns, 2)      
      plt.imshow(mask[0])  
      plt.axis('off') 
      plt.title("mask")        
  else:
    # display([sample_image, sample_mask,
             # create_mask(model.predict(sample_image[tf.newaxis, ...]))])
      i=0
      for image, mask in val_ds.take(1):
        for i in range(4):
          image = image
          mask = mask
          # fig.add_subplot(rows, columns, j) 
          j=0     
          axes_disp[i,j].imshow(image[i])  
          axes_disp[i,j].axis('off') 
          axes_disp[i,j].set_title("image")  

          # fig.add_subplot(rows, columns, j)   
          j+=1   
          axes_disp[i,j].imshow(create_mask(mask[i]))  
          axes_disp[i,j].axis('off') 
          axes_disp[i,j].set_title("mask")      

          # fig.add_subplot(rows, columns, j) 
          j+=1     
          print("shape of fed image", image[i][tf.newaxis,...].shape)
          prediction = model.predict(image[i][tf.newaxis,...])
          print("shape of prediction", prediction.shape)

          axes_disp[i,j].imshow(create_mask(prediction[0]))  
          axes_disp[i,j].axis('off') 
          axes_disp[i,j].set_title("prediction")  
      plt.show(block=block)
      plt.pause(0.5)
      # plt.close()

show_predictions()
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    # clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


callbacks = [
    keras.callbacks.ModelCheckpoint("savedModels/"+label+".h5", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25),
    DisplayCallback()
]
# Train the model, doing validation at the end of each epoch.
epochs = 5000
start = time()
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
print("training time is " + str(time()-start)+" seconds")

import matplotlib.pyplot as plt
import matplotlib
show_predictions(block=True)
# matplotlib.use('Agg') 


print(history.history.keys())

loss, val_loss, accuracy, val_accuracy = [], [], [], []

loss = loss + history.history['loss']
val_loss = val_loss + history.history['val_loss']
accuracy = accuracy + history.history['io_u']
val_accuracy = val_accuracy + history.history['val_io_u']

fig, ax = plt.subplots()
ax.plot(accuracy,label = 'train')
ax.plot(val_accuracy,label = 'test')
ax.set_title('io_u')
ax.legend(loc='lower right')
fig.savefig('io_u'+label+'.png')

fig, ax = plt.subplots()
ax.plot(loss,label = 'train')
ax.plot(val_loss,label = 'test')
ax.set_title('Loss')
ax.legend(loc='upper right')
fig.savefig('loss'+label+'.png')