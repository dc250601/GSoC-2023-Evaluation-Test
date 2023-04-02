import torch
from e2cnn import gspaces
from e2cnn import nn

class model(torch.nn.Module):
    """Model: An equivariant model adapted and inspired from the VGG-19 architecture 
    Original model had 64,128,256,512,512 features
    This model has 8,16,32,64,64 features
    """
    def __init__(self, channels = 1, N=4,image_shape = (128,128), group = "cyclic"):
        super(model,self).__init__()
        self.channels = channels
        self.N = N
        if group== "cyclic":
            self.r2_act = gspaces.Rot2dOnR2(N=self.N)
        elif group == "dihyderal":
            self.r2_act = gspaces.FlipRot2dOnR2(N=self.N)
        else:
            print(f"Group {group} not found")
        self.block1 = nn.SequentialModule(
            nn.MaskModule(nn.FieldType(self.r2_act, self.channels*[self.r2_act.trivial_repr]),
                          image_shape[0],
                          margin=1
                         ),
            #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, self.channels*[self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]),
                                           sigma=0.66,
                                           stride=2)
        )
        
        self.block2 = nn.SequentialModule(
         #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 8*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                                           sigma=0.66,
                                           stride=2)   
        )
        
        self.block3 = nn.SequentialModule(
             #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]), inplace=True),
             #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]), inplace=True),
             #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]),
                                           sigma=0.66,
                                           stride=2)   
        )
        
        self.block4 = nn.SequentialModule(
             #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True),
             #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True),
             #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                                           sigma=0.66,
                                           stride=2)   
        )
        
        self.block5 = nn.SequentialModule(
             #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True),
             #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True),
             #------------------------------------------------------------------------------
            nn.R2Conv(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                      kernel_size=3,
                      padding=1,
                      bias=False
                     ),
            nn.InnerBatchNorm(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])),
            nn.ReLU(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]), inplace=True),
            #------------------------------------------------------------------------------
            nn.PointwiseAvgPoolAntialiased(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                                           sigma=0.66,
                                           stride=2)   
        )
        self.final_pool = nn.PointwiseMaxPoolAntialiased(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]),
                                                         sigma=0.66,
                                                         stride=1,
                                                         kernel_size = 4)
        self.gpool = nn.GroupPooling(nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr]))
        self.fc = torch.nn.Linear(64,1)
    def forward(self, x):
        x = nn.GeometricTensor(x,
                               nn.FieldType(self.r2_act, self.channels*[self.r2_act.trivial_repr]))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.final_pool(x)
        x = self.gpool(x)
        x = x.tensor
        x = x.squeeze()
        x = self.fc(x)
        x = x.squeeze()
        return x