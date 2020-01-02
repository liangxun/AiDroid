import torch
from torch import nn
from torch.nn import functional as F
import torchsnooper


class GoogLeNet(nn.Module):
    """参考AiDroid论文中Figure 6.
    """
    def __init__(self, dropout=0.4):
        super(GoogLeNet, self).__init__()
        self.drop_rate = dropout

        self.layer1_conv = nn.Conv2d(1, 6, 3)
        self.layer1_bn = nn.BatchNorm2d(6)
        self.layer1_pool = nn.MaxPool2d(kernel_size=3)

        self.layer2_conv = nn.Conv2d(6, 16, 1)
        self.layer2_bn = nn.BatchNorm2d(16)

        self.layer3_conv = nn.Conv2d(16, 32, 3)
        self.layer3_bn = nn.BatchNorm2d(32)
        self.layer3_pool = nn.MaxPool2d(kernel_size=3)

        self.layer4_inception = Inception(32, 64)

        self.layer5_pool = nn.MaxPool2d(kernel_size=6)

        self.fc = nn.Linear(64, 2)

    # @torchsnooper.snoop()
    def forward(self, input): # input's shape=(*, 1, 64, 64)
        if input.dim()==3:
            input.unsqueeze_(1)
        layer1_output = self.layer1_pool(self.layer1_bn(F.relu(self.layer1_conv(input)))) # shape=( *, 6, 20, 20)
        layer2_output = self.layer2_bn(F.relu(self.layer2_conv(layer1_output))) # shape=(*, 16, 18, 18)
        layer3_output = self.layer3_pool(self.layer3_bn(F.relu(self.layer3_conv(layer2_output))))  # shape=(*, 32, 6, 6)
        layer4_output = self.layer4_inception(layer3_output) # shape=(*, 64, 6, 6)
        layer5_output = self.layer5_pool(layer4_output).squeeze()  # shape=(*, 64)
        output = self.fc(F.dropout(layer5_output, p=self.drop_rate))
        return output


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        out_channels自己决定,这里每个branch设置了16个通道，所以输出一共为64个通道
        """
        super(Inception, self).__init__()
        self.branch1_conv1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch2_conv1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2_conv3x3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.branch3_conv1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3_conv5x5 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)

        self.branch4_maxpool3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
    
    def forward(self, input):
        branch1 = F.relu(self.branch1_conv1x1(input))
        branch2 = F.relu(self.branch2_conv3x3(F.relu(self.branch2_conv1x1(input))))
        branch3 = F.relu(self.branch3_conv5x5(F.relu(self.branch3_conv1x1(input))))
        branch4 = F.relu(self.branch4_conv1x1(self.branch4_maxpool3x3(input)))
        output = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return output
