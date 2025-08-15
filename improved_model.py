import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class ImprovedIdentify(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.features = nn.Sequential(
            # 300 * 300 * 3
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # 150 * 150 * 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            # 75 * 75 * 64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.4),

            #  37 * 37 * 128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict(self, x, batch_size=32):
        self.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        y_pred = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                outputs = self.forward(inputs)
                preds = torch.argmax(outputs, dim=1)
                y_pred.extend(preds.cpu().numpy())

        return np.array(y_pred)