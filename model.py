import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable


class ConvolutionalNN(nn.Module):
    '''subclass of pytorch's nn.Module class to create a NN with two convolutional layers followed by four fully connected
    ones. Max-pooling is performed after the conv layers and ReLU is used as an activation funciton. The last layer is
    a softmax funcitno.
    '''

    def __init__(self):
        '''initializes the NN.'''

        super(ConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 63)  # 2 * 26 characters + 10 numbers + 1 trash
        self.fc4 = nn.Linear(63, 37)  # 26 characters + 10 numbers + 1 trash

    def forward(self, x):
        '''defines the forward pass through the NN.'''

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return F.log_softmax(out)
