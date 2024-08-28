# Food Image Classification App
### 0151_Image_Classification_Using_PyTorch_46_Labels_50_Percent
#### 0151_46_Labels_50_Percent

## Table of contents
* [About](#about)
* [General Information](#general-information)
* [Libraries Used](#libraries-used)
* [Dataset Information](#dataset-information)
* [Model Creation](#model-creation)
* [Side Notes](#side-notes)
* [Resources](#resources)
* [Other](#other)


## About
Image Classification App for 46 food labels.

<br>

## General Information
Image classification web app that can predict 46 different types of food given an input image.
Model Label increment improvement. Increment from 10 to 46 labels with an accuracy of around 70%.
The app uses a Feature Extractor (EfficientNet) to predict whether an image belongs to one of the trained food labels.
To see all of the labels the model can predict see the class_names text file. Some of the food labels this model can predict are:
- apple_pie
- breakfast_burrito
- club_sandwich
- donuts
- french_fries
- grilled_salmon
- hamburger
- hot_dog
- ice_cream
- lasagna
- pizza
- ramen
- samosa
- steak
- sushi
- tacos
- waffles

(Please see 'data/class_names.txt' to see all the labels this model can be used for)

<br>

## Libraries Used
Some of the libraries used for this project:
- Pytorch
- Gradio
- Pillow


<br>

## Dataset Information

EfficientNetB2 Feature Extractor was trained on [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
[Food-101 dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html) was downloaded From PyTorch. A percentage of the entire dataset was randomly selected and gathered for specific labels to retrain the model.

<br>

## Model Creation
This iteration consisted on creating a model with 46 labels using 50% of the whole dataset, which consists on around 1000 images per label.

##### Splits

| | Train | Test |
|--|--|--|
| | 80 % | 20 % |


- Example Images: Performed a last test with some random sample images gathered from the test set. Some of those images were used for the example dataset


<br>

## Side Notes
For more information about this project please see 'main_report.pdf'. This reports contains more information regarding the scope of this project along with the decisions taken, the project implementation and some of its results.

<br>

## Resources

##### Some of the resources used:

[EfficientNet](https://pytorch.org/vision/main/models/efficientnet.html)

[Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

[Food-101 Dataset on PyTorch](https://pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)

@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}


## Other

Local Virtual Environment Information
- Environment Name: venv_0151_Deployment_310_002
- Python Version: 3.10.12
