import torch
import torchvision
import gradio as gr
import os

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

import pandas as pd

#import torch
#import torchvision
#from torch import nn
#def create_effnetb2_model(num_classes:int=10,
#                          seed:int=42):
#    ## Create EffNetB2 pretrained weights, transforms and model
#    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
#    transforms = weights.transforms()
#    model = torchvision.models.efficientnet_b2(weights=weights)
#
#    ## Freeze all layers in the base model
#    for param in model.parameters():
#        param.requires_grad = False
#
#    ## Change classifier head with random seed for reproducibility
#    torch.manual_seed(seed)
#    model.classifier = torch.nn.Sequential(
#        torch.nn.Dropout(p=0.3, inplace=True),
#        torch.nn.Linear(in_features=1408, out_features=num_classes)
#    )
#
#    return model, transforms


## Setup class names
class_names_path = './class_names.txt'
with open(class_names_path, 'r') as f:
    food_class_names_loaded = [food.strip() for food in f.readlines()]

len(food_class_names_loaded)

class_names = food_class_names_loaded

class_names_df_02 = pd.DataFrame({"Food": class_names})
class_names_df_02 = class_names_df_02.reset_index()
class_names_df_02["index"] = class_names_df_02["index"]+1
class_names_df_02.rename(columns={"index": "No"}, inplace=True)

#class_names_df = pd.DataFrame({
    #"Food": ['apple_pie', 'breakfast_burrito', 'caesar_salad', 'carrot_cake', #'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'churros', #'club_sandwich', 'cup_cakes', 'deviled_eggs', 'donuts', 'eggs_benedict', #'filet_mignon', 'french_fries', 'french_toast', 'fried_rice', 'garlic_bread', #'greek_salad', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_dog', #'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'macaroni_and_cheese', #'nachos', 'omelette', 'onion_rings', 'oysters', 'paella', 'pancakes', 'pizza', #'ramen', 'ravioli', 'risotto', 'samosa', 'spaghetti_bolognese', #'spaghetti_carbonara', 'steak', 'sushi', 'tacos', 'waffles']
    #})


### 2. Model and transforms preparation
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=46)

## Load and save weights
#effnetb2.load_state_dict(torch.load("./effnetb2_feature_extractor_food_46_50_percent_001_epochs_10_001.pth", map_location=torch.device("cpu")))
effnetb2.load_state_dict(torch.load("./effnetb2_feature_extractor_food_46_50_percent_001_epochs_10_001.pth", map_location=torch.device("cpu"), weights_only=True))

## Warning: FutureWarning: You are using `torch.load` with `weights_only=False` -> Uses Pickle module
# Recommendation on using `weights_only=True` for any use case where you don't have full control of the loaded file. To prevent malicious code


### 3. Predict function
def predict(img) -> Tuple[Dict, float]:
  start_time = timer()

  # Transform the input image for use with EffNetB2
  img = effnetb2_transforms(img).unsqueeze(0) # unsqueeze = add batch dimension on 0th index

  # Put model into eval mode, make prediction
  effnetb2.eval()
  with torch.inference_mode():
    # Pass transformed image through the model and turn the prediction logits into probaiblities
    pred_probs = torch.softmax(effnetb2(img), dim=1)

  # Create a prediction label and prediction probability dictionary
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  end_time = timer()
  pred_time = round(end_time - start_time, 4)

  return pred_labels_and_probs, pred_time



### 4. Gradio App
title = 'Image Classification For 46 Food Labels'
description = "Using a feature extractor computer vision model to classify several food images."
article = "* Model improvement."

long_article = """This iteration uses an EfficientNetB2 Feature Extractor Model.
The improvement was done by increasing the amount of labels the model can predict. From 10 to 46 labels. The model's accuracy performance is around 70%."""


## Create example list -> Note: List of lists
example_list = [["examples/" + example] for example in os.listdir("./examples/")]

## Create Gradio Demo App
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=10, label='Predictions'),
                             gr.Number(label="Prediction time (s)")
                             ],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

with gr.Blocks() as demo:
   gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=5, label='Predictions'),
                             gr.Number(label="Prediction time (s)")
                             ],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)
   
   with gr.Accordion("Open for more information.", open=False):
     gr.Markdown(long_article)
   
   #with gr.Accordion("Open to see all the labels this model can predict.", open=False):
     #gr.Dataframe(class_names_df)
   
   with gr.Accordion("Open to see all the food this model can predict.", open=False):
     gr.DataFrame(class_names_df_02)
    

## Launch Demo App
demo.launch()

