from typing import Dict

import numpy as np
import torch
from torch import nn as nn

from torch.autograd import Variable

import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax

import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def conv_output_dim(dimension, padding, dilation, kernel_size, stride
):
    r"""Calculates the output height and width based on the input
    height and width to the convolution layer.

    ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    """
    assert len(dimension) == 2
    out_dimension = []
    for i in range(len(dimension)):
        out_dimension.append(
            int(
                np.floor(
                    (
                            (
                                    dimension[i]
                                    + 2 * padding[i]
                                    - dilation[i] * (kernel_size[i] - 1)
                                    - 1
                            )
                            / stride[i]
                    )
                    + 1
                )
            )
        )
    return tuple(out_dimension)

EPS = 1e-6

def layer_init(cnn):
    for layer in cnn:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                layer.weight, nn.init.calculate_gain("relu")
            )
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)


class AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN for processing audio spectrogram features

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """
    @property
    def output_size(self):
        return self.rnn_dim

    def __init__(self, input_size, output_size):
        super(AudioCNN, self).__init__()
        self._n_input_audio = input_size[-1]
        cnn_dims = np.array(input_size[:2], dtype=np.float32)

        if cnn_dims[0] < 35 or cnn_dims[1] < 35:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.rnn_dim = output_size

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )
        self.get_orientation_embedding()

        layer_init(self.cnn)


    def get_orientation_embedding(self):
        self.orientation_embedding = torch.zeros(1, 24, 32)
        for i in range(24):
            alpha = math.pi * 2 * i / 24
            alpha = torch.ones(16) * alpha
            embedding = torch.cat([torch.sin(alpha), torch.cos(alpha)])
            self.orientation_embedding[0, i] = embedding

    def forward(self, observations):
        '''
            cnn_input: [BATCH x CHANNEL x HEIGHT X WIDTH]
        '''
        if 'spectrogram' in observations:
            cnn_input = observations['spectrogram'].float()
            b, h, w, c = cnn_input.shape
            cnn_input = cnn_input.view(b, h, w, c)
            cnn_input = cnn_input.permute(0, 3, 1, 2)
            output = self.cnn(cnn_input).view(b, -1)
            # output = torch.cat([output, self.orientation_embedding.to(cnn_input.device).expand(b, 24, 32)], -1)
            return output
        else:
            return None
