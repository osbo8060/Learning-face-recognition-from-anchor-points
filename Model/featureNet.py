import torch
import torch.nn as nn
import torch.nn.functional as F

###
# Def Inception_Stem The stem from Incetion-ResNet-v1
# input_shape - The shape of our input image (height, width, channels)
# Returns - The tensor after passing through the stem block.
###
class Inception_stem(nn.Module):
    def __init__(self, input_channels=3):
        super(Inception_stem, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv4 = nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.conv1(x))  # Output: 149x149x32
        x = self.relu(self.conv2(x))  # Output: 147x147x32
        x = self.relu(self.conv3(x))  # Output: 147x147x64
        x = self.maxpool(x)           # Output: 73x73x64
        x = self.relu(self.conv4(x))  # Output: 73x73x80
        x = self.relu(self.conv5(x))  # Output: 71x71x192
        x = self.relu(self.conv6(x))  # Output: 35x35x256
        return x



### 
# InceptionResNetABlock: Defines one Inception-ResNet-A block of size 35x35
# filters - The amount of filters for the convolutions
# Returns - The tensor after the A block
###
class InceptionResNet_A_Block(nn.Module):
    def __init__(self, in_channel, filters=32):
        super(InceptionResNet_A_Block, self).__init__()

        # Branch 1
        self.branch1 = nn.Conv2d(in_channel, filters, kernel_size=1, stride=1, padding=0)

        # Branch 2
        self.branch2_1 = nn.Conv2d(in_channel, filters, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        # Branch 3
        self.branch3_1 = nn.Conv2d(in_channel, filters, kernel_size=1, stride=1, padding=0)
        self.branch3_2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.branch3_3 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.reduced_conv = nn.Conv2d(filters*3, in_channel, kernel_size=1, stride=1, padding=0) # (*3 because 3 dimensions)
        self.relu = nn.ReLU()

    def forward(self, x):

        # Branch 1
        branch_1 = self.relu(self.branch1(x))

        # Branch 2
        branch_2 = self.relu(self.branch2_1(x))
        branch_2 = self.relu(self.branch2_2(branch_2))

        # Branch 3
        branch_3 = self.relu(self.branch3_1(x))
        branch_3 = self.relu(self.branch3_2(branch_3))
        branch_3 = self.relu(self.branch3_3(branch_3))

        # Concatenate all branches along the channel dimension
        mixed = torch.cat([branch_1, branch_2, branch_3], dim=1)

        reduced = self.reduced_conv(mixed)
        output = torch.add(x, reduced)
        output = self.relu(output)

        return output
    


### 
# InceptionResNet_B_Block: Defines one Inception-ResNet-B block of size 17x17
# filters - The amount of filters for the convolutions
# in_channels - The amount of input channels
# Returns - The tensor after the B block
### 
class InceptionResNet_B_Block(nn.Module):
    def __init__(self, in_channels, filters=32):
        super(InceptionResNet_B_Block, self).__init__()

        # Branch 1
        self.branch1 = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0)

        # Branch 2
        self.branch2_1 = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = nn.Conv2d(filters, filters, kernel_size=(1,7), stride=1, padding=(0,3))
        self.branch2_3 = nn.Conv2d(filters, filters, kernel_size=(7,1), stride=1, padding=(3,0))

        self.reduced_conv = nn.Conv2d(filters*2, in_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Branch 1
        branch_1 = self.relu(self.branch1(x))

        # Branch 2
        branch_2 = self.relu(self.branch2_1(x))
        branch_2 = self.relu(self.branch2_2(branch_2))

        # Concatenate
        mixed = torch.cat([branch_1, branch_2], dim=1)

        reduced = self.reduced_conv(mixed)
        output = torch.add(x, reduced)
        output = self.relu(output)

        return output
    
### 
# InceptionResNet_C_Block: Defines one Inception-ResNet-C block of size 8x8
# in_channels - The amount of input channels
# filters - The amount of filters for the convolutions
# Returns - The tensor after the C block
###
class InceptionResNet_C_Block(nn.Module):
    def __init__(self, in_channels=1792, filters=32):
        super(InceptionResNet_C_Block, self).__init__()

        # Branch 1
        self.branch_1 = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0)

        # Branch 2 NOTE: Kanske ska ha större filterstorlek?
        self.branch_2_1 = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, padding=0)
        self.branch_2_2 = nn.Conv2d(filters, filters, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch_2_3 = nn.Conv2d(filters, filters, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # 1x1 Convolution to reduce dimensions
        self.reduced = nn.Conv2d(filters*2, 1792, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Branch 1
        branch_1 = self.branch_1(x)

        # Branch 2
        branch_2 = self.branch_2_1(x)
        branch_2 = self.branch_2_2(branch_2)
        branch_2 = self.branch_2_3(branch_2)

        # Concatenate
        mixed = torch.cat([branch_1, branch_2], dim=1)
        reduced = self.reduced(mixed)
        output = x + reduced
        output = F.relu(output)

        return output
    
# NOTE: Fattade inte helt, fanns ingen klar arkitektur som de gjorde, så jag frågade GPT hur jag skulle implementera. Detta är deras med 3 branches. Verkar rimligt, men vet ej om vi endast vill ha ett pooling layer istället
class ReductionA(nn.Module):
    def __init__(self, in_channels, k=192, l=192, m=256, n=384):
        super(ReductionA, self).__init__()

        # Branch 1: 3x3 Max Pooling
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Branch 2: 3x3 Convolution with stride 2
        self.branch2 = nn.Conv2d(in_channels, n, kernel_size=3, stride=2, padding=0)

        # Branch 3: 1x1 -> 3x3 -> 3x3 Convolutions with stride 2
        self.branch3_1 = nn.Conv2d(in_channels, k, kernel_size=1, stride=1, padding=0)
        self.branch3_2 = nn.Conv2d(k, l, kernel_size=3, stride=1, padding=1)
        self.branch3_3 = nn.Conv2d(l, m, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        # Process each branch
        branch1 = self.branch1(x)  # Max pooling
        branch2 = self.branch2(x)  # Convolution
        branch3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))  # Triple convolution

        # Concatenate branches along the channel dimension
        return torch.cat([branch1, branch2, branch3], dim=1)

class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()

        # Branch 1
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Branch 2
        self.branch2_1 = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)

        # Branch 3
        self.branch3_1 = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.branch3_2 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.branch3_3 = nn.Conv2d(192, 320, kernel_size=3, stride=2, padding=0)

        # Branch 4
        self.branch4_1 = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.branch4_2 = nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=0)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2_2(self.branch2_1(x))
        branch3 = self.branch3_3(self.branch3_2(self.branch3_1(x)))
        branch4 = self.branch4_2(self.branch4_1(x))

        # Concatenate
        output = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        return output

### 
# FullyConnectedLayer: Adds a fully connected layer with dropout
# in_channels - The amount of input channels
# feature_dim: d from article
# dropout_rate: The dropout rate
# returns - the tensor after the FCL
###
class FullyConnectedLayer(nn.Module):
    def __init__(self, in_channels, feature_dim, dropout_rate=0.8):
        super(FullyConnectedLayer, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, feature_dim)

    def forward(self, x):
        x = self.global_avg_pool(x)

        # Flatten [batch_size, channels, 1, 1] to [batch_size, channels]
        x = torch.flatten(x, 1) 
        x = self.dropout(x)
        x = self.fc(x)

        return x

###
# feature_net: Implements the feature net from BioMetricNet: https://arxiv.org/pdf/2008.06021
# input_shape: The shape of the images as inputs, in the paper they use 299, 299, 3
# feature_dim: The size of the output feature vector (d), see paper
# dropout_rate: The dropout rate (reduces overfitting by randomly dropping neurons during training, ensuring the network generalizes well)
###
class FeatureNet(nn.Module):
    def __init__(self, feature_dim=512, dropout_rate=0.8):
        super(FeatureNet, self).__init__()

        self.inception_stem = Inception_stem()
        self.inception_resnet_a = InceptionResNet_A_Block(in_channel=256,filters=32)  
        self.reduction_a = ReductionA(in_channels=256, k=192, l=192, m=256, n=384)
        self.inception_resnet_b = InceptionResNet_B_Block(in_channels=896, filters=32)  
        self.reduction_b = ReductionB(in_channels=896)
        self.inception_resnet_c = InceptionResNet_C_Block(in_channels=1792, filters=32)  
        self.fc_layer = FullyConnectedLayer(in_channels=1792, feature_dim=feature_dim, dropout_rate=dropout_rate)

    def forward(self, x):

        x = self.inception_stem(x)
        
        for _ in range(5):
            x = self.inception_resnet_a(x)

        x = self.reduction_a(x)
        
        for _ in range(10):
            x = self.inception_resnet_b(x)
        
        x = self.reduction_b(x)

        for _ in range(5):
            x = self.inception_resnet_c(x)
        
        x = self.fc_layer(x)
        
        return x

