import timm
import torch.nn as nn

def create_model(num_classes=19, pretrained=True):
    model = timm.create_model("resnet50", pretrained=pretrained)
    in_features = model.get_classifier().in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
        nn.Sigmoid()
    )
    return model