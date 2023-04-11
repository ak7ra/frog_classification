import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()                 
        
        ### Convolutional Layers ###
        self.convolutional = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2), 
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        
            # Layer 2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),    
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        
            # Layer 3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),      
            nn.BatchNorm2d(384), 
            nn.ReLU(inplace=True),
        
            # Layer 4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),    
            nn.BatchNorm2d(384), 
            nn.ReLU(inplace=True),
        
            # Layer 5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        ### Linear layers ###
        self.classifier = nn.Sequential(
        
            # Feature map to 6x6x256 size
            nn.AdaptiveAvgPool2d((6, 6)),
            # Flatten for fully connected layers
            nn.Flatten(),
            
            # Layer 1
            nn.Dropout(p=0.1),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
        
            # Layer 2
            nn.Dropout(p=0.1),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
        
            # Layer 3
            nn.Dropout(p=0.1),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        
            # Layer 4
            nn.Linear(in_features=4096, out_features=num_classes))
        
    # Forward 
    def forward(self, x):
        x = self.convolutional(x)
        x = self.classifier(x)
        return x