import torch
import torchvision
#from torch import nn

def create_effnetb2_model(num_classes:int=10,
                          seed:int=42):
    ## Create EffNetB2 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    ## Freeze all layers in the base model
    for param in model.parameters():
        param.requires_grad = False

    ## Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features=1408, out_features=num_classes)
    )

    return model, transforms