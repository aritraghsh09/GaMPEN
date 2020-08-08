import torch
import torch.nn as nn
import torch.nn.functional as F

from groupy.gconv.pytorch_gconv.splitgconv2d import P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

class GGT(nn.Module):
    """
    Galaxy Group Transformer
    """


    def __init__(self):
        super(GGT, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 96, kernel_size=9),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(True)
        )

        # Fully-connected regression network (predicts 3 * 2 affine matrix)
        self.fc_loc = nn.Sequential(
            nn.Linear(96 * 34 * 34, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        ident = [1, 0, 0, 0, 1, 0]
        self.fc_loc[2].bias.data.copy_(torch.tensor(ident, dtype=torch.float))

        # Featurizer blocks -- these need to be separate so that we can
        # intersperse plane_group_spatial_max_pooling operations in `forward`
        self.featurize1 = nn.Sequential(
            P4MConvZ2(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True)
        )
        self.featurize2 = nn.Sequential(
            P4MConvP4M(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.featurize3 = nn.Sequential(
            P4MConvP4M(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            P4MConvP4M(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            P4MConvP4M(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((6, 6))

        # Fully-connected regression
        self.regress = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )


    def spatial_transform(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 96 * 34 * 34)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x


    def forward(self, x):
        x = self.spatial_transform(x)
        x = self.featurize1(x)
        x = plane_group_spatial_max_pooling(x, ksize=3, stride=2)
        x = self.featurize2(x)
        x = plane_group_spatial_max_pooling(x, ksize=3, stride=2)
        x = self.featurize3(x)
        x = plane_group_spatial_max_pooling(x, ksize=3, stride=2)
        x = x.view(x.size()[0], x.size()[1], x.size()[2], -1)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.regress(x)

        return x