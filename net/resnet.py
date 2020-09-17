import net.network as network
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.functional as F
def ResK(dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=True):
    """
    :param dim: INPUT = OUTPUT
    :param padding_type: reflect replicate zero
    :param norm_layer: instance or batch
    :param activation: relu
    :param use_dropout:  default use
    :return: resnetblock class, access the .conv_block to get the Sequential model
    """
    return ResnetBlock(dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=True)

class ResnetBlock(nn.Module):
    def __init__(self,dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=True):
        """
        Rk denotes a residual block that contains TWO 3*3
        convolutional layers with the same number of filters on both
        layers.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self,dim,padding_type, norm_layer, activation, use_dropout):
        """
        :param dim: same input and output
        :param padding_type: fill or not fill & reflect or replicate or none
        :param norm_layer: instance or batch
        :param activation:
        :param use_dropout:
        :return: nn.Sequential
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == 'replicate':
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += network.cNsN_K(input_channels=dim,stride=1,k=dim,N=3,padding=p,norm=norm_layer,activation=activation)
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))
            conv_block += conv_block[:-2]
        else:
            conv_block += conv_block[:-1]
        return nn.Sequential(*conv_block)

    def forward(self,x):

        the_output = x + self.conv_block(x)
        return the_output
        # https://www.cnblogs.com/wuliytTaotao/p/9560205.html

class SpadeResBlock(nn.Module):
    def __init__(self, ni, nf,opt):
        super(SpadeResBlock, self).__init__()
        self.spade_bn0 = network.SpadeBN(ni)
        self.conv0 = nn.Conv2d(ni, nf, kernel_size=3, padding=1)
        self.spade_bn1 = network.SpadeBN(nf)
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.spade_skip = network.SpadeBN(ni)
        self.conv_skip = nn.Conv2d(ni, nf, kernel_size=3, padding=1)
        if opt.use_spectral:
            self.conv0 = spectral_norm(self.conv0)
            self.conv1 = spectral_norm(self.conv1)
            self.conv_skip = spectral_norm(self.conv_skip)

    def forward(self, input, segmap):
        # skip seq = spade -> relu -> conv
        skip_features = self.conv_skip(F.relu(self.spade_skip(input, segmap)))
        # spade0 -> relu -> conv0 -> spade1 -> relu -> conv1
        features = self.conv0(F.relu(self.spade_bn0(input, segmap)))
        features = self.conv1(F.relu(self.spade_bn1(features, segmap)))
        return skip_features + features




