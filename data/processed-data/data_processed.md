# Processed Data Folder

## Introduction
This folder contains the data that has been:
- Converted into ECG charts, stored as PNG files.
- Packaged into LMDB datasets - one containing 224x224 images, another containing 300x300 images.

<br>

## Image Configs
Two separate datasets were created:
- Config 1: Contains one ECG chart layout.
- Config 2: Contains two ECG chart layouts.


<br>

## Folder Structure
Each config is stored in its own directory, with the following structure:

```
 processed-data
 ┃
 ┣ config1
 ┃ ┣ test
 ┃ ┃ ┣ img_efficientnet.lmdb
 ┃ ┃ ┣ img_resnet.lmdb
 ┃ ┃ ┣ png-files
 ┃ ┣ train
 ┃ ┃ ┣ img_efficientnet.lmdb
 ┃ ┃ ┣ img_resnet.lmdb
 ┃ ┃ ┣ png-files
 ┃
 ┣ config1
 ┃ ┣ test
 ┃ ┃ ┣ img_efficientnet.lmdb
 ┃ ┃ ┣ img_resnet.lmdb
 ┃ ┃ ┣ png-files
 ┃ ┣ train
 ┃ ┃ ┣ img_efficientnet.lmdb
 ┃ ┃ ┣ img_resnet.lmdb
 ┃ ┃ ┣ png-files
 ```

 Note that the `img_resnet.lmdb` represents the dataset containing 224x224 images, while the `img_efficientnet.lmdb` contains 300x300 images. The naming conventions were unfortunately due to initial experiments using ResNet-18 and EfficientNet-B3 models.