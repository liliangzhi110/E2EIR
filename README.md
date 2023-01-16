# Multimodal image fusion framework for end-to-end remote sensing image registration



![image-20230109174009382](image-20230109174009382.png)

## Contents

1. Dataset and Data generation
2. Training


## Dataset and Data generation

For training, validating, and testing the proposed network,we employ the PS-RGB(RGB), Multispectral(MS), andSAR-Intensity(SAR) datasets from the SpaceNet [32] dataset.

In addition to the dataset itself, we have provided the scripts for data generation. To generate data,using the .py file in the **E2EIR/generate_affine_pre_data/generate_train_data/** folder

```Shell
E2EIR/generate_affine_pre_data/generate_train_data/
```
Or you can use the already generated training data

```Shell

dateset1: 0.23m resolution https://drive.google.com/drive/folders/1xyH2P1TRsRd9u2oXFGbYQ9zyQ4Ewu5U4?usp=sharing

dataset2: 3.75m resolution https://drive.google.com/drive/folders/14kMVwVdvZ9YEFrwieqqMW1Fkn5Ne381S?usp=sharing

dataset3: 30m resolution https://drive.google.com/drive/folders/1KL1wQ9-1oFthXH9YaCCNJ7KuRXozxu7M?usp=sharing
```


## Training
Dataset is generated. You can use the dataset by running:

```Shell
python train.py
```




