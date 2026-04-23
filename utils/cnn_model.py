import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, base_channels=16, dropout_feat=0.2, dropout=0.3):
        super().__init__()
        c = base_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),             # 69 -> 34

            nn.Conv2d(c, 2*c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),             # 34 -> 17

            nn.Conv2d(2*c, 4*c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),             # 17 -> 8
            nn.Dropout(dropout_feat),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*c * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)            # logits
        )

    def forward(self, x):
        return self.classifier(self.features(x))