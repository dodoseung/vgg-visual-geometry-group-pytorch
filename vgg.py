import torch.nn.functional as F
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, input_img=[3, 224, 224], output_channel=10, num_blocks=18):
        super(VGG, self).__init__()
        # Condition
        self.num_blocks = num_blocks
        assert self.num_blocks in [11, 13, 16, 19], 'Please set the number of layers as 11, 13, 16, 19'
        
        self.input_img = input_img
        self.output_channel = output_channel

        if self.num_blocks == 11:
            self.vgg = nn.Sequential(
                nn.Conv2d(self.input_img[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            
        elif self.num_blocks == 13:
            self.vgg = nn.Sequential(
                nn.Conv2d(self.input_img[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            
        elif self.num_blocks == 16:
            self.vgg = nn.Sequential(
                nn.Conv2d(self.input_img[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        
        elif self.num_blocks == 19:
            self.vgg = nn.Sequential(
                nn.Conv2d(self.input_img[0], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(256), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(512), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            
        # Set the output layer
        self.img_size = [int(self.input_img[1]/32), int(self.input_img[2]/32)]
        self.output = nn.Sequential(
                nn.Linear(512 * self.img_size[0] * self.img_size[1], 4096), nn.ReLU(),
                nn.Linear(4096, 4096), nn.ReLU(),
                nn.Linear(4096, self.output_channel), nn.ReLU())


    def forward(self, x):
        # VGG
        x = self.vgg(x)
        
        # Output
        x = x.view(-1, 512 * self.img_size[0] * self.img_size[1])
        x = self.output(x)
        x = F.softmax(x, dim=-1)
        
        return x