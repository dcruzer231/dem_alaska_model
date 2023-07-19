import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import geopandas as gpd
import numpy as np
from PIL import Image
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


# datadir = "/media/dan/dataBackup1/DEM_Site/Images/DEM_Slope_Aspect_Composite_Raster/"
# dataname = "Brw_Phantom_DEM_2022_TIN2Raster_NAD83_0.008m_Composite_3Bands.tif"
datadir = "/home/dan/dem_site_project/calmsite/Brw_CALM_DEM_Slope_Aspect_3Band_Composite/"
dataname = "Brw_CALM_Subset_RGB_DEM_Slope_Aspect_3Band_Composite_190802.tif"


# labeldir = "/media/dan/dataBackup/DEM_Site/Labels/Labels_Polygon2Raster/"
# labeldir = "./"
# label = "Brw_DEM_2Class_Labels_4.tif"
labeldir= "/home/dan/dem_site_project/calmsite/Brw_CALM_Labels/"
label = "Brw_CALM_Subset_labels_Polygon2Raster.tif"

rgbdir = "/home/dan/dem_site_project/calmsite/Brw_CALM_RGB_Ortho/"
rgbname ="Brw_CALM_Subset_RGB_Ortho_190802.tif"

y = tifffile.imread(labeldir+label)
x = tifffile.imread(datadir+dataname)
rgb =tifffile.imread(rgbdir+rgbname)
x[x<0] = 0
# print((img/(img.max()+1)).max())
# plt.imshow(img/(img.max()+1))
plt.figure(1)
print("image",x.shape, "label",y.shape)
print("x max  is", x.max())
print("x min  is", x.min())
print("x mean is",np.mean(x))
print("x std is",np.std(x))



print("x[0] max  is", x[:,:,0].max())
print("x[0] min  is", x[:,:,0].min())
print("x[0] mean is",np.mean(x[:,:,0]))
print("x[0] std is",np.std(x[:,:,0]))

print("x[1] max  is", x[:,:,1].max())
print("x[1] min  is", x[:,:,1].min())
print("x[1] mean is",np.mean(x[:,:,1]))
print("x[1] std is",np.std(x[:,:,1]))
print(x[:,:,1].flatten())


print("x[2] max  is", x[:,:,2].max())
print("x[2] min  is", x[:,:,2].min())
print("x[2] mean is",np.mean(x[:,:,2]))
print("x[2] std is",np.std(x[:,:,2]))


# exit()

xnorm = np.copy(x)
xnorm[:,:,0]  = MinMaxScaler().fit_transform(x[:,:,0])#stats.zscore(x[:,:,0])#/= x[:,:,0].max()
xnorm[:,:,1]  = MinMaxScaler().fit_transform(x[:,:,1])#stats.zscore(x[:,:,1])#/= x[:,:,1].max()
xnorm[:,:,2]  = MinMaxScaler().fit_transform(x[:,:,2])#stats.zscore(x[:,:,2])#/= x[:,:,2].max()
xnorm = np.nan_to_num(xnorm)
# xnorm = xnorm[:,:,::-1]
plt.imshow(xnorm)
plt.figure(2)
plt.imshow(y)
yr = y/y.max()



def make3Band(label,flip=False):
    lableim = np.zeros(shape=(label.shape+(3,)))
    if flip:
        lableim[:,:,0] = (label == 2) * 255
        lableim[:,:,1] = (label == 1) * 255
        lableim[:,:,2] = (label == 3) * 255
    else:    
        lableim[:,:,0] = (label == 1) * 255
        lableim[:,:,1] = (label == 2) * 255
        lableim[:,:,2] = (label == 3) * 255
    return lableim


def shift_label(label, img,offsety, offsetx):
    canvas = np.zeros_like(img[:,:,0])
    ys = label.shape[0]
    xs = label.shape[1]

    canvas[offsety:ys+offsety, offsetx:xs+offsetx] = y
    canvas[canvas == 0] = 3
    ydim = np.expand_dims(canvas,axis=2)

    return ydim

def resize_label(label,img,mode=None):
    # ydim = np.expand_dims(label,axis=2)
    # plt.imshow(ydim)
    # plt.show()
    # exit()
    if mode:
        label_image = Image.fromarray(label,mode=mode)
    else:
        label_image = Image.fromarray(label)
    label_image = label_image.resize((img.shape[1],img.shape[0]))
    y_array = np.array(label_image)
    # plt.imshow(y_array)
    return y_array

plt.figure(3)
shifted_label = shift_label(yr,xnorm,65,50)
print(shifted_label.shape)
# labelcanvas = np.zeros_like(xnorm)

# labelcanvas[:,:,0] = (canvas == 1) * 255
# labelcanvas[:,:,1] = (canvas == 2)* 255
# labelcanvas[:,:,2] = (canvas == 3)* 255
# labelcanvas = labelcanvas.astype(np.uint8)
plt.imshow(shifted_label)

# plt.imshow(canvas3d)

# plt.imshow(y3band)
# plt.show()
yn = resize_label(yr,rgb)

yn= np.round(yn*3)

# z = xnorm+(ydim)#/ydim.max())
# z[:,:,0]  /= z[:,:,0].max()
# z[:,:,1]  /= z[:,:,1].max()
# z[:,:,2]  /= z[:,:,2].max()
# z[z<0] = 0
plt.imshow(yn)

plt.figure(4)
# print(x.shape, "\n", labelcanvas.shape)

# im = Image.fromarray(labelcanvas)
# im.save("calm_label.png")

y3band = make3Band(yn,True)
plt.imshow(y3band)
# np.save("calm_label",y3band)

xnorm *= 255
xnorm = xnorm.astype(np.uint8)
im = Image.fromarray(xnorm)
# im.save("calm_data.png")

# plt.figure(5)
# plt.imshow(x[:,:,0])
# plt.figure(6)
# plt.imshow(x[:,:,1])
# plt.figure(7)
# plt.imshow(x[:,:,2])

plt.figure(5)
print("type of RGB",rgb.dtype)
rgb = rgb.astype(np.uint8)
xnorm_resized = resize_label(xnorm,rgb)
xnorm_resized = xnorm_resized.astype(np.float64)
xnorm_resized /= 255
plt.imshow(xnorm_resized)

plt.figure(6)
plt.imshow(rgb)

print(rgb.shape, xnorm_resized.shape   )
x6band = np.concatenate((rgb,xnorm_resized),axis=2)
print(x6band.shape)
np.save("calm_6band",x6band)
plt.imsave("calm_label.png",y3band.astype(np.uint8))




plt.show()  