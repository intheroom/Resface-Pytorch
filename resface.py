import torch
import torch.nn as nn
import torch.nn.functional as F

class resface_pre(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(resface_pre, self).__init__()
        self.output_channels = output_channels
        self.conv = nn.Conv2d(input_channels, output_channels, 3, 2, 1)
        self.relu = nn.PReLU(output_channels)
    def forward(self, x):
        return self.relu(self.conv(x))

class resface_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(resface_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1)
        self.relu1 = nn.PReLU(output_channels)
        self.relu2 = nn.PReLU(output_channels)
    def forward(self, x):
        net = self.relu1(self.conv1(x))
        net = self.relu2(self.conv2(net))
        return x + net

class resface20(nn.Module):
    def __init__(self, embedding_size=128, p=0.5):
        super(resface20, self).__init__()
        self.pre1 = resface_pre(3, 64)
        self.block1_1 = resface_block(64, 64)
        self.pre2 = resface_pre(64, 128)
        self.block2_1 = resface_block(128, 128)
        self.block2_2 = resface_block(128, 128)
        self.pre3 = resface_pre(128, 256)
        self.block3_1 = resface_block(256, 256)
        self.block3_2 = resface_block(256, 256)
        self.block3_3 = resface_block(256, 256)
        self.block3_4 = resface_block(256, 256)
        self.pre4 = resface_pre(256, 512)
        self.block4_1 = resface_block(512, 512)
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(512 * 7 * 6, embedding_size)
    def forward(self, x):
        net = self.pre1(x)
        net = self.block1_1(net)
        net = self.pre2(net)
        net = self.block2_2(self.block2_1(net))
        net = self.pre3(net)
        net = self.block3_4(self.block3_3(self.block3_2(self.block3_1(net))))
        net = self.pre4(net)
        net = self.block4_1(net)
        net = net.view(net.size(0), -1)
        net = self.dropout(net)
        net = self.fc(net)
        return net

class resface36(nn.Module):
    def __init__(self, embedding_size=128, p=0.5):
        super(resface36, self).__init__()
        self.pre1 = resface_pre(3, 64)
        self.block1_1 = resface_block(64, 64)
        self.block1_2 = resface_block(64, 64)
        self.pre2 = resface_pre(64, 128)
        self.block2_1 = resface_block(128, 128)
        self.block2_2 = resface_block(128, 128)
        self.block2_3 = resface_block(128, 128)
        self.block2_4 = resface_block(128, 128)
        self.pre3 = resface_pre(128, 256)
        self.block3_1 = resface_block(256, 256)
        self.block3_2 = resface_block(256, 256)
        self.block3_3 = resface_block(256, 256)
        self.block3_4 = resface_block(256, 256)
        self.block3_5 = resface_block(256, 256)
        self.block3_6 = resface_block(256, 256)
        self.block3_7 = resface_block(256, 256)
        self.block3_8 = resface_block(256, 256)
        self.pre4 = resface_pre(256, 512)
        self.block4_1 = resface_block(512, 512)
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(512 * 7 * 6, embedding_size)
    def forward(self, x):
        net = self.pre1(x)
        net = self.block1_2(self.block1_1(net))
        net = self.pre2(net)
        net = self.block2_4(self.block2_3(self.block2_2(self.block2_1(net))))
        net = self.pre3(net)
        net = self.block3_4(self.block3_3(self.block3_2(self.block3_1(net))))
        net = self.block3_8(self.block3_7(self.block3_6(self.block3_5(net))))
        net = self.pre4(net)
        net = self.block4_1(net)
        net = net.view(net.size(0), -1)
        net = self.dropout(net)
        net = self.fc(net)
        return net