import torch
from torch import nn

class model(torch.nn.Module):
    """Model: A non equivariant model adapted and inspired from the VGG-19 architecture 
    Original model had 64,128,256,512,512 features
    This model has 16,32,64,128,128 features
    """
    def __init__(self, channels = 1,image_shape = (128,128)):
        super(model,self).__init__()
        self.channels = channels
        
        self.block1 = nn.Sequential(
            #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=channels,
                      out_channels=16,
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.AvgPool2d(stride=2,kernel_size=2)
        )
        
        self.block2 = nn.Sequential(
         #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.AvgPool2d(stride=2,kernel_size=2)
        )
        
        self.block3 = nn.Sequential(
            #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=32,
                      out_channels=64,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=64,
                      out_channels=64,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
             #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=64,
                      out_channels=64,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
             #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=64,
                      out_channels=64,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.AvgPool2d(stride=2,kernel_size=2)
        )
        
        self.block4 = nn.Sequential(
            #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=64,
                      out_channels=128,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=128,
                      out_channels=128,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
             #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=128,
                      out_channels=128,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
             #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=128,
                      out_channels=128,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.AvgPool2d(stride=2,kernel_size=2)
        )
        
        self.block5 = nn.Sequential(
            #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=128,
                      out_channels=128,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=128,
                      out_channels=128,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
             #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=128,
                      out_channels=128,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
             #------------------------------------------------------------------------------
            nn.Conv2d(in_channels=128,
                      out_channels=128,kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #------------------------------------------------------------------------------
            nn.AvgPool2d(stride=2,kernel_size=2)
        )
        
        self.final_pool = nn.MaxPool2d(kernel_size=4,stride=1)
        self.fc = torch.nn.Linear(128,1)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.final_pool(x)
        x = x.squeeze()
        x = self.fc(x)
        x = x.squeeze()
        return x