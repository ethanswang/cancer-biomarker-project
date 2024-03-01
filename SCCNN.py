import torch
import torch.nn as nn
import torch.nn.functional as F

class SCCNN(nn.Module):
    def __init__(self, height, width, num_classes):
        super(SCCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 1 channel input
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adjust the size based on your input dimensions after pooling
        def get_conv_output_size(size, kernel_size=3, padding=1, stride=1, pool=2):
            return ((size - kernel_size + 2 * padding) // stride + 1) // pool
        
        conv_height = get_conv_output_size(get_conv_output_size(get_conv_output_size(height)))
        conv_width = get_conv_output_size(get_conv_output_size(get_conv_output_size(width)))
        
        self.fc1 = nn.Linear(64 * conv_height * conv_width, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
