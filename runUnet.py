import numpy as np 
import matplotlib.pyplot as plt
import pathlib
# from unet import get_model
from patchify import patchify, unpatchify
from tensorflow.keras.models import load_model
from loss_functions import dice_loss
import segmentation_models as sm

def findstepsize(w,p):
    for i in range(p,2,-1):
        # print((8149-p)%i)
        if (w-p)%i == 0:
            return i


# label = "resnet34_calm_6band_dicesloss_512crop_flipaugment"
label = "resnet34_calm_5band_dicesloss_512crop_flipaugment_float32"

savedModel = "savedModels/"+label+".h5"

model = load_model(savedModel,custom_objects={"dice_loss": dice_loss, "iou_score":sm.metrics.IOUScore(smooth=1e-02)})
chanNum = (3,)
skipChannel = [5]
skpch = [i for i in np.arange(6) if i not in skipChannel]


datadir = pathlib.Path(r"/home/dan/dem_site_project")
dataFile = pathlib.Path(r"data_6band.npy")

img = np.load(datadir / dataFile).astype(np.float32)[...,skpch]

# patches = patchify(img, (512,512), step=512)

#slices images based on patch size.  If the patch slice does not match the image size evenly then padding will be added
#returns an array of slices and a tuple of length-width shapes for the original image and the image with padding
def slice(image,slice_shape=512):
    from math import ceil
    #size is of 7566, 8143
    oldshape = image.shape[:-1]
    newshape = (ceil(image.shape[0]/slice_shape) * slice_shape, ceil(image.shape[1]/slice_shape) * slice_shape)
    canvas = np.zeros(newshape+(image.shape[-1],))
    canvas[:image.shape[0],:image.shape[1],...] = image
    crops = []
    for i in range(int(canvas.shape[0]/(slice_shape))):
        for j in range(int(canvas.shape[1]/(slice_shape))):
            crop = canvas[slice_shape * i:slice_shape *(i+1),slice_shape*j:slice_shape*(j+1),]
            crops.append(crop)
    del canvas
    return np.array(crops), (oldshape,newshape)

#stitches together all the slices back to original shape. requires the shapes tuple returned from slice. crops is expected to be the same dimensions as the output of slice.
def stitch(crops,shapes,slice_shape=512):
    oldshape,newshape = shapes
    channel = crops.shape[-1]
    canvas = np.zeros(newshape+(channel,))
    # patched = np.zeros(image.shape[:2]+(channel,))
    # print("input",image.shape,"patched",patched.shape)
    for i in range(int(canvas.shape[0]/(slice_shape))):
        for j in range(int(canvas.shape[1]/(slice_shape))):
            # l = j 
            # m = int(img.shape[0]/(slice_shape)) + i
            # plt.imshow(img[i,j])
            canvas[slice_shape * i:slice_shape *(i+1),slice_shape*j:slice_shape*(j+1),] = crops[i * int(canvas.shape[1]/(slice_shape)) + j]
            # patched[256* factor * i:slice_shape *(i+1),slice_shape*j:slice_shape*(j+1),] = crops[i * int(img.shape[1]/(slice_shape)) + j]
    return canvas[:oldshape[0],:oldshape[1],...]


crops,shapes = slice(img,512)
# crops = crops[:,:,:,:3]
print(crops.shape)
predicts = model.predict(crops)
print(predicts.shape)

patched = stitch(predicts,shapes,512)

prediction = np.rint(patched)



plt.imshow(np.rint(patched))
plt.figure(2)
plt.imshow(img[:,:,:3])
# plt.figure(3)
# plt.imshow(img[:,:,3:])
plt.show()
