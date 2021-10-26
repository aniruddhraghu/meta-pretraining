import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal



def conv5x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)

def conv9x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4)

def conv15x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=stride, padding=7)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm1d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv15x1(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv15x1(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
#         out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_outputs=5, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=nn.BatchNorm1d):

        super(ResNet, self).__init__()
        
        self.num_outputs = num_outputs

        self._norm_layer = norm_layer

        self.inplanes = 32

        self.conv1 = nn.Conv1d(12, self.inplanes, kernel_size=15, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.proj = nn.Sequential(nn.Linear(256, 256, bias=False),
                               nn.ReLU(inplace=True), nn.Linear(256, 128, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
#             print("Got to downsample place")
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def logits(self,x):
        BS, L, C = x.shape
        x = x.transpose(1,2)
        return self._forward_impl(x)

    def forward(self, x):
        BS, L, C = x.shape
        x = x.transpose(1,2)
        feature = self._forward_impl(x)
        out = self.proj(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


def _resnet(arch, block, layers, pretrained2, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def ecg_simclr_resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)




from torch.distributions.normal import Normal
from scipy.ndimage import gaussian_filter1d

class MultitaskHead(torch.nn.Module):
    def __init__(self, feats, num_classes=5):
        super(MultitaskHead, self).__init__()
        self.fc_pi = torch.nn.Linear(feats, num_classes)

    def forward(self, x):
        out_pi = self.fc_pi(x)
        return out_pi


#### Adapted from https://github.com/voxelmorph/voxelmorph
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)
#         print("grid shape", grid.shape)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(src.shape) == 3:
            src = src.unsqueeze(-1).repeat(1,1,1,2)
            new_locs = new_locs.unsqueeze(-1).repeat(1,1,1,2)
        
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        
        samp = F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        return samp.squeeze(2) #samp[:,:,:,0]


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
#         print("vecshape", vec.shape)
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec



class ResizeTransformTime(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, sf, ndims):
        super().__init__()
        self.sf = sf
        self.mode = 'linear'

    def forward(self, x):
        factor = self.sf
        if factor < 1:
            x = F.interpolate(x, align_corners=False, scale_factor=factor, mode=self.mode, recompute_scale_factor=False)
            x = factor * x
        elif factor > 1:
            x = factor * x
            x = F.interpolate(x, align_corners=False, scale_factor=factor, mode=self.mode, recompute_scale_factor=False)
        return x



from scipy.ndimage import gaussian_filter1d


# per example magnitude
class RandWarpAugLearnExMag(nn.Module):
    def __init__(self, inshape, int_steps = 5, int_downsize = 4, flow_mag=4, smooth_size = 25):

        super().__init__()
        
        ndims=1
        self.inshape=inshape
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransformTime(1/int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransformTime(int_downsize, ndims) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [inshape[0]//int_downsize]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
        
        # set up smoothing filter
        self.flow_mag = torch.nn.parameter.Parameter(torch.Tensor([float(flow_mag)]))
        self.smooth_size= smooth_size
        self.smooth_pad = smooth_centre = (smooth_size-1)//2
        smooth_kernel = np.zeros(smooth_size)
        smooth_kernel[smooth_centre] = 1
        filt = gaussian_filter1d(smooth_kernel, smooth_centre).astype(np.float32)
        self.smooth_kernel = torch.from_numpy(filt)
        
        self.net = nn.Sequential(nn.Conv1d(12, 32, 15,stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 15, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 15, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 15, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten())
        
        self.flow_mag_layer = nn.Linear(32,1)

    def forward(self, source):
        BS, L, C = source.shape
        source = source.transpose(1,2)
        x=source
        fm = 2*torch.sigmoid(self.flow_mag_layer(self.net(x)))
        
        fm_std = 100*(self.flow_mag**2)

        flow_field = fm.view(BS, 1, 1) * fm_std*torch.randn(x.shape[0], 1, self.inshape[0]).to(x.device)
        
        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        # DO SOME SMOOTHING OF THE FLOW FIELD HERE.       
        pos_flow = F.conv1d(pos_flow, self.smooth_kernel.view(1,1,self.smooth_size).to(x.device), padding=self.smooth_pad, stride=1)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        return y_source.transpose(1,2)
    
    