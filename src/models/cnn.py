import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.5):
        super().__init__()
        # blok 1
        self.conv1     = nn.Conv2d(1,  32, 3, padding=1)
        self.bn1       = nn.BatchNorm2d(32)
        # blok 2
        self.conv2     = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2       = nn.BatchNorm2d(64)
        # pooling + dropout
        self.pool      = nn.MaxPool2d(2,2)
        self.drop_conv = nn.Dropout2d(dropout_p)
        # blok 3
        self.conv3     = nn.Conv2d(64,128, 3, padding=1)
        self.bn3       = nn.BatchNorm2d(128)
        # blok 4
        self.conv4     = nn.Conv2d(128,128, 3, padding=1)
        self.bn4       = nn.BatchNorm2d(128)
        # pooling + dropout
        self.pool2     = nn.MaxPool2d(2,2)
        self.drop_conv2= nn.Dropout2d(dropout_p)
        # classifier
        self.fc1       = nn.Linear(128*7*7, 256)
        self.drop_fc   = nn.Dropout(dropout_p)
        self.fc2       = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop_conv(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = self.drop_conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x)
