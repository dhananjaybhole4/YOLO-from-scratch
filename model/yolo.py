import torch
import torch.nn as nn

import yaml

from pathlib import Path

class ConvBlock(nn.Module):
    """This class returns a 2d Convolutional layer used in neural network

    Args:
        input_channels: in_channels for conv layer
        output_channels: out_channels for conv layer
        kernal: kernal size for conv layer
        stride: stride size for conv layer
        padding: padding size for conv layer
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
    def forward(self, x: torch.Tensor):
        return self.relu(self.batchnorm(self.conv(x)))
    

class YOLOv1(nn.Module):
    """Model architecture similar to the YOLOv1

    Args:
        nn (_type_): _description_
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        self.conv_architecture_config_path = Path.cwd()/"config"/"architecture.yaml"
        # convolution block
        self.conv_block = nn.Sequential(*self.get_conv_block(self.conv_architecture_config_path))
        # connecting block
        self.conn_block = self.get_conn_block(**kwargs)

    # extracting information from yaml file
    def get_conv_block(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

            # building conv, maxpool layers from config file
            layers = []
            for layer in cfg["conv_architecture"]:
                if layer["type"] == "conv2d":
                    layers.append(ConvBlock(in_channels = layer["args"]["in_channels"],
                                            out_channels = layer["args"]["out_channels"],
                                            kernel_size = layer["args"]["kernel_size"],
                                            stride = layer["args"]["stride"],
                                            padding = layer["args"]["padding"]))
                elif layer["type"] == "maxpool":
                    layers.append(nn.MaxPool2d(kernel_size = layer["args"]["kernel_size"],
                                            stride = layer["args"]["stride"]))
                elif layer["type"] == "repetative_conv":
                    for i in range(layer["repetation"]):
                        for j in layer["content"]:
                            layers.append(ConvBlock(in_channels = j["args"]["in_channels"],
                                                    out_channels = j["args"]["out_channels"],
                                                    kernel_size = j["args"]["kernel_size"],
                                                    stride = j["args"]["stride"],
                                                    padding = j["args"]["padding"]))
        return layers
    
    def get_conn_block(self, split_size = 7, num_boxes = 2, num_classes = 11):
        S, B, C = split_size, num_boxes, num_classes
        conn_block = nn.Sequential(nn.Flatten(),
                                        nn.Linear(1024*7*7, 16384),
                                        nn.LeakyReLU(0.1),
                                        nn.Linear(16384, S * S * (C + B * 5))
                                        )
        return conn_block
        
    def forward(self, x: torch.Tensor):
        return self.conn_block(self.conv_block(x))

