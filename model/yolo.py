import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """This class returns a 2d Convolutional layer used in neural network

    Args:
        input_channels: in_channels for conv layer
        output_channels: out_channels for conv layer
        kernal: kernal size for conv layer
        stride: stride size for conv layer
        padding: padding size for conv layer
    """
    def __init__(self, input_channels,
                 output_channels,
                 kernel,
                 stride,
                 padding):
        super().__init__()
        self.convblock = nn.Conv2d(in_channels = input_channels,
                              out_channels = output_channels,
                              kernel_size = kernel,
                              stride = stride,
                              padding = padding)
    def forward(self, x: torch.Tensor):
        return self.convblock(x)
    

