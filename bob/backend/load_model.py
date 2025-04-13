import os
import json
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Model definition moved from app.py
class ResNet50FineTuneMultiHead(nn.Module):
    def __init__(self, num_genres, num_styles, dropout_rate=0.0):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Freeze parameters for base layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Unfreeze the last 4 children layers
        for child in list(self.resnet.children())[-4:]:
            for param in child.parameters():
                param.requires_grad = True

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the final fc layer

        # Define separate classifiers for genre and style
        self.genre_classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_genres)
        )
        self.style_classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_styles)
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.genre_classifier(features), self.style_classifier(features)

def load_model(app):
    config_path = os.path.join(app.root_path, 'models', 'best_config.json')
    model_path = os.path.join(app.root_path, 'models', 'best_resnet50_finetune_multihhead.pth')

    # Load configuration parameters
    with open(config_path, "r") as f:
        best_config = json.load(f)

    num_genres = 11
    num_styles = 11

    model_instance = ResNet50FineTuneMultiHead(num_genres, num_styles, best_config["dropout_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance.to(device)

    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()

    print("Classifier loaded and ready.")
    return model_instance
