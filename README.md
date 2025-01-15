# Semantic Segmentation of DEM data using U-Net architecture
The goal of this project was to segment drone imagery of Alaskan landscape.  The landscape was to be segmented into valleys and hills.  A U-net model was trained on expert labeled data using dice-loss.  In the end the output was not good enough to be used for analysis and the project was halted.

### RGB representation of the data
![alt text](https://github.com/dcruzer231/dem_alaska_model/blob/main/images/datasite_RGB.png)
### The DEM data visualized
![alt text](https://github.com/dcruzer231/dem_alaska_model/blob/main/images/data_visualization_compressed.jpg)
### The label data
![alt text](https://github.com/dcruzer231/dem_alaska_model/blob/main/images/label2.png)
### output using dice-loss
![alt text](https://github.com/dcruzer231/dem_alaska_model/blob/main/images/resnet34_calm_5band_dicesloss_512crop_flipaugment_float32_prediction.png)
