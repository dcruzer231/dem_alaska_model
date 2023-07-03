import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import geopandas as gpd
import numpy as np
from PIL import Image
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


datadir = "/media/dan/dataBackup/DEM_Site/Images/DEM_Slope_Aspect_Composite_Raster/"
dataname = "Brw_Phantom_DEM_2022_TIN2Raster_NAD83_0.008m_Composite_3Bands.tif"

# labeldir = "/media/dan/dataBackup/DEM_Site/Labels/Labels_Polygon2Raster/"
labeldir = "./"
label = "Brw_DEM_2Class_Labels_4.tif"



y = tifffile.imread(labeldir+label)
x = tifffile.imread(datadir+dataname)
x[x<0] = 0
# print((img/(img.max()+1)).max())
# plt.imshow(img/(img.max()+1))
plt.figure(1)
# print("x max  is", x.max())
# print("x min  is", x.min())
# print("x mean is",np.mean(x))
# print("x std is",np.std(x))



# print("x[0] max  is", x[:,:,0].max())
# print("x[0] min  is", x[:,:,0].min())
# print("x[0] mean is",np.mean(x[:,:,0]))
# print("x[0] std is",np.std(x[:,:,0]))

# print("x[1] max  is", x[:,:,1].max())
# print("x[1] min  is", x[:,:,1].min())
# print("x[1] mean is",np.mean(x[:,:,1]))
# print("x[1] std is",np.std(x[:,:,1]))
# print(x[:,:,1].flatten())


# print("x[2] max  is", x[:,:,2].max())
# print("x[2] min  is", x[:,:,2].min())
# print("x[2] mean is",np.mean(x[:,:,2]))
# print("x[2] std is",np.std(x[:,:,2]))


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





plt.figure(3)

offset = 65
offsetx = 50
canvas = np.zeros_like(xnorm[:,:,0])
canvas[offset:7501+offset, offsetx:8054+offsetx] = y
canvas[canvas == 0] = 3
ydim = np.expand_dims(yr,axis=2)
canvas3d = np.copy(xnorm)

# canvas3d[offset:(7212+offset), offsetx:8054+offsetx,]= xnorm[offset:(7212+offset), offsetx:8054+offsetx,:]+(ydim)#/ydim.max())
canvas3d[:,:,0]  /= canvas3d[:,:,0].max()
canvas3d[:,:,1]  /= canvas3d[:,:,1].max()
canvas3d[:,:,2]  /= canvas3d[:,:,2].max()
# plt.imshow(canvas3d)

labelcanvas = np.zeros_like(xnorm)

labelcanvas[:,:,0] = (canvas == 1) * 255
labelcanvas[:,:,1] = (canvas == 2)* 255
labelcanvas[:,:,2] = (canvas == 3)* 255
labelcanvas = labelcanvas.astype(np.uint8)

# yim = Image.fromarray(yr)
# yim = yim.resize((x.shape[1],x.shape[0]))
# yn = np.array(yim)
# plt.imshow(yn)
ydim = np.expand_dims(canvas,axis=2)
z = xnorm+(ydim)#/ydim.max())
z[:,:,0]  /= z[:,:,0].max()
z[:,:,1]  /= z[:,:,1].max()
z[:,:,2]  /= z[:,:,2].max()
# z[z<0] = 0
plt.imshow(z*2)

plt.figure(4)
print(x.shape, "\n", labelcanvas.shape)

im = Image.fromarray(labelcanvas)
im.save("label2.png")
plt.imshow(canvas)
np.save("label2",canvas)

xnorm *= 255
xnorm = xnorm.astype(np.uint8)
im = Image.fromarray(xnorm)
im.save("data2.png")

# plt.figure(5)
# plt.imshow(x[:,:,0])
# plt.figure(6)
# plt.imshow(x[:,:,1])
# plt.figure(7)
# plt.imshow(x[:,:,2])


plt.show()	