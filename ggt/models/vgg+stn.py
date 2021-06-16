import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from ggt.utils.model_utils import get_output_shape


class vgg_w_stn(nn.Module):
    
    def __init__(self, cutout_size, channels, n_out=1, pretrained=True):
        super(vgg_w_stn, self).__init__()
        self.cutout_size = cutout_size
        self.channels = channels
        self.expected_input_shape = (
            1,
            self.channels,
            self.cutout_size,
            self.cutout_size,
        )
        self.n_out = n_out
        self.pretrained = pretrained
        
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=11),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 96, kernel_size=9),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(True),
        )
        
        # Calculate the output size of the localization network
        self.ln_out_shape = get_output_shape(
            self.localization, self.expected_input_shape
        )

        # Calculate the input size of the upcoming FC layer
        self.fc_in_size = torch.prod(torch.tensor(self.ln_out_shape[-3:]))

        # Fully-connected regression network (predicts 3 * 2 affine matrix)
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_in_size, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        ident = [1, 0, 0, 0, 1, 0]
        self.fc_loc[2].bias.data.copy_(torch.tensor(ident, dtype=torch.float))

        
        #Featurizer -- VGG
        self.vgg = models.vgg16(pretrained=self.pretrained)
        self.vgg.classifier[6] = nn.Linear(4096, self.n_out)
        
        
    def spatial_transform(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_in_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x
        
    def forward(self, x):
        x = self.spatial_transform(x)
        x = self.vgg(x)

        return x