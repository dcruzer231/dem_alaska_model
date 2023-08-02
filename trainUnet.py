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
import numpy as np
from skimage.transform import resize


# tf.random.set_seed(1337)
AUTOTUNE = tf.data.AUTOTUNE

import os
#uncomment to force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

input_dir = pathlib.Path("/home/dan/dem_site_project/calm_datacrop_6band_minmaxscaler/")
target_dir =  pathlib.Path("/home/dan/dem_site_project/calm_labelcrop/")

BACKBONE = 'resnet34'
label = "resnet34_calm_6band_dicesloss_512crop_flipaugment"
img_size = (512, 512)
#the background class and the particle class
num_classes = 3 
batch_size = 4

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".npy")
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

def normalize_numpy(x):
  x[:,:,:3] /= x[:,:,:3].max()
  # for i in range(3,x.shape[-1]):
    # x[:,:,i] = x[:,:,i]/x[:,:,i].max()
  return x

for input_path, target_path in zip(input_img_paths[:4],target_img_paths[:4]):
    print(input_path, "|", target_path)
input_img_paths = [normalize_numpy(np.load(img)) for img in input_img_paths]
target_img_paths = [plt.imread(img) for img in target_img_paths]


dataset = tf.data.Dataset.from_tensor_slices((input_img_paths,target_img_paths))

length = tf.data.experimental.cardinality(dataset).numpy()

val_size = int(length * 0.1)
train_ds = dataset.skip(val_size)
val_ds = dataset.take(val_size)

print("train size",tf.data.experimental.cardinality(train_ds).numpy())
print("validation size",tf.data.experimental.cardinality(val_ds).numpy())

def decode_imgs(img,mask):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.read_file(img)
  img = tf.io.decode_raw(img, tf.float64)
  # tf.numpy_function(np.load, img, tf.float64)
  # img = tf.io.decode_png(img,channels=6)
  # img = np.load(img.decode())
  img = img/255
  mask = tf.io.read_file(mask)
  mask = tf.io.decode_png(mask,channels=3)
  img = tf.py_function(resize, (img,img_size+(6,)),tf.float64)

  # Resize the image to the desired size
  return img, tf.image.resize(mask, img_size)

# for input_path, target_path in train_ds.take(10):
    # print(input_path, "|", target_path)

# train_ds = train_ds.map(decode_imgs, num_parallel_calls=AUTOTUNE)
# val_ds = val_ds.map(decode_imgs, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.shuffle(length,seed=1337)
val_ds = val_ds.shuffle(length,seed=1337)

train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# train_ds = train_ds.cache()
val_ds = val_ds.cache()

# train_ds = train_ds.repeat() #repeat forever
# val_ds = val_ds.repeat() #repeat forever

def flipx_img(x,y, p=0.5):
  if  tf.random.uniform([]) < p:
    x = tf.reverse(x,[2])
    y = tf.reverse(y,[2])
  else:
    x
    y
  return x, y

def flipx(factor=0.5):
  return tf.keras.layers.Lambda(lambda x: flipx_img(x, factor))

flipx = flipx()

def flipy_img(x,y, p=0.5):
  if  tf.random.uniform([]) < p:
    x = tf.reverse(x,[1])
    y = tf.reverse(y,[1])
  else:
    x
    y
  return x, y

def flipy(factor=0.5):
  return tf.keras.layers.Lambda(lambda x: flipy_img(x, factor))

# image, label = next(iter(train_ds))
# aug_image, aug_label = flipy_img(image,label)
# image = image[0,:,:,:3]
# label = label[0,:,:]
# aug_image = aug_image[0,:,:,:3]
# aug_label = aug_label[0,:,:]

# fig, axis = plt.subplots(2, 2)
# axis[0,0].imshow(image)
# axis[0,1].imshow(aug_image)
# axis[1,0].imshow(label)
# axis[1,1].imshow(aug_label)

# plt.show()


flipy = flipy()

# def augment(image_label, seed):
  # image, label = image_label
def augment(image, label):  

  print('image', image) # these lines is in the augment function, result below
  print('type', type(image))
  print('label', label) # these lines is in the augment function, result below
  print('type', type(label))  
  # image, label = resize_and_rescale(image, label)
  # image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
  # Make a new seed.
  # new_seed = tf.random.split(seed, num=1)[0, :]
  # Random crop back to the original size.
  # image = tf.image.stateless_random_crop(
      # image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness.
  # image = tf.image.stateless_random_brightness(
      # image, max_delta=0.5, seed=new_seed)
  # image = tf.clip_by_value(image, 0, 1)
  image,label = flipx_img(image,label)
  image,label = flipy_img(image,label)
  return image, label

# aug_ds = flipx(train_ds)#train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
# aug_ds = flipy(aug_ds)#train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Create a generator.
rng = tf.random.Generator.from_seed(123, alg='philox')

# Create a wrapper function for updating seeds.
def f(x, y):
  seed = rng.make_seeds(2)[0]
  print("x in f is", x)
  print("rank of x", tf.rank(x))
  image, label = augment((x, y), seed)
  return image, label

for image, mask in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("mask shape: ", mask.numpy().shape)


#print a partial list of found images
print("Number of samples:", len(input_img_paths), len(target_img_paths))

aug_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
# aug_ds = aug_ds.prefetch(buffer_size=AUTOTUNE)








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
model = sm.Unet(BACKBONE, classes=num_classes, input_shape=img_size+(6,), encoder_weights=None)
# model = get_model(img_size,3)
# model.summary()
print("exit callbacks")
# model.compile(optimizer="adam", loss=sm.losses.DiceLoss(), metrics=[sm.metrics.IOUScore(), tf.keras.metrics.IoU(num_classes=3, target_class_ids=[0,1])])
model.compile(optimizer="adam", loss=sm.losses.DiceLoss(), metrics=[sm.metrics.IOUScore(smooth=1e-02), tf.keras.metrics.Accuracy()])


def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask

def create_maskrgb(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  # pred_mask = pred_mask[..., tf.newaxis]
  canvas = np.zeros(img_size+(3,))
  canvas[:,:,0] = (pred_mask ==0 )
  canvas[:,:,1] = (pred_mask ==1 )
  canvas[:,:,2] = (pred_mask ==2 )

  return canvas

rows = 4
columns = 3
fig_disp,axes_disp = plt.subplots(rows,columns)
def show_predictions(dataset=None, num=1,block=False):
  if dataset:
    for image, mask in dataset.take(num):
      showimg = image[:,:,:,:3]
      pred_mask = model.predict(image)
      fig = plt.figure(figsize=(10, 7))

      # display([image[0], mask[0], create_mask(pred_mask)])
      # showing image
      plt.imshow(showimg[0])  
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

          image3ch = image[:,:,:,:3]
          print("validation show image is of shape",image3ch.shape)
          mask = mask
          # fig.add_subplot(rows, columns, j) 
          j=0     
          axes_disp[i,j].imshow(image3ch[i])  
          axes_disp[i,j].axis('off') 
          axes_disp[i,j].set_title("image")  

          # fig.add_subplot(rows, columns, j)   
          j+=1   
          axes_disp[i,j].imshow(create_maskrgb(mask[i]))  
          axes_disp[i,j].axis('off') 
          axes_disp[i,j].set_title("mask")      

          # fig.add_subplot(rows, columns, j) 
          j+=1     
          print("shape of fed image", image[i][tf.newaxis,...].shape)
          prediction = model.predict(image[i][tf.newaxis,...])
          print("shape of prediction", prediction.shape)

          axes_disp[i,j].imshow(create_maskrgb(prediction[0]))  
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
history = model.fit(aug_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
print("training time is " + str(time()-start)+" seconds")

import matplotlib.pyplot as plt
import matplotlib
show_predictions(block=True)
# matplotlib.use('Agg') 


print(history.history.keys())

# loss, val_loss, accuracy, val_accuracy = [], [], [], []

loss = loss + history.history['loss']
val_loss = val_loss + history.history['val_loss']
accuracy = accuracy + history.history['iou_score']
val_accuracy = val_accuracy + history.history['val_iou_score']

fig, ax = plt.subplots()
ax.plot(accuracy,label = 'train')
ax.plot(val_accuracy,label = 'test')
ax.set_title('iou_score')
ax.legend(loc='lower right')
fig.savefig('iou_score'+label+'.png')

fig, ax = plt.subplots()
ax.plot(loss,label = 'train')
ax.plot(val_loss,label = 'test')
ax.set_title('Loss')
ax.legend(loc='upper right')
fig.savefig('loss'+label+'.png')