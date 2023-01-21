import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import sqrt


def normc_fn(m):
    """
    This function multiplies the weights of a pytorch linear layer by a small
    number so that outputs early in training are close to zero, which means 
    that gradients are larger in magnitude. This means a richer gradient signal
    is propagated back and speeds up learning (probably).
    """
    if m.__class__.__name__.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def create_layers(layer_fn, input_dim, layer_sizes):
    """
    This function creates a pytorch modulelist and appends
    pytorch modules like nn.Linear or nn.LSTMCell passed
    in through the layer_fn argument, using the sizes
    specified in the layer_sizes list.
    """
    ret = nn.ModuleList()
    ret += [layer_fn(input_dim, layer_sizes[0])]
    for i in range(len(layer_sizes) - 1):
        ret += [layer_fn(layer_sizes[i], layer_sizes[i + 1])]
    return ret


class Net(nn.Module):
    """
    The base class which all policy networks inherit from. It includes methods
    for normalizing states.
    """

    def __init__(self):
        super(Net, self).__init__()
        # nn.Module.__init__(self)
        self.is_recurrent = False

        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1

        self.env_name = None

        self.calculate_norm = False

    def normalize_state(self, state, update=True):
        """
        Use Welford's algorithm to normalize a state, and optionally update the statistics
        for normalizing states using the new state, online.
        """
        state = torch.Tensor(state)

        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1))
            self.welford_state_mean_diff = torch.ones(state.size(-1))

        if update:
            if len(state.size()) == 1:  # if we get a single state vector
                state_old = self.welford_state_mean
                self.welford_state_mean += (state - state_old) / self.welford_state_n
                self.welford_state_mean_diff += (state - state_old) * (state - state_old)
                self.welford_state_n += 1
            else:
                raise RuntimeError  # this really should not happen
        return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)

    def copy_normalizer_stats(self, net):
        self.welford_state_mean = net.welford_state_mean
        self.welford_state_mean_diff = net.welford_state_mean_diff
        self.welford_state_n = net.welford_state_n

    def initialize_parameters(self):
        self.apply(normc_fn)
        if hasattr(self, 'network_out'):
            self.network_out.weight.data.mul_(0.01)


class FF_Base(Net):
    """
    The base class for feedforward networks.
    """

    def __init__(self, in_dim, layers, nonlinearity):
        super(FF_Base, self).__init__()
        self.layers = create_layers(nn.Linear, in_dim, layers)
        self.nonlinearity = nonlinearity

    def _base_forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = self.nonlinearity(layer(x))
        return x


class LSTM_Base(Net):
    """
    The base class for LSTM networks.
    """

    def __init__(self, in_dim, layers):
        super(LSTM_Base, self).__init__()
        self.layers = create_layers(nn.LSTMCell, in_dim, layers)

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]

    def _base_forward(self, x):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))

            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]

                y.append(x_t)
            x = torch.stack([x_t for x_t in y])
        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)
        return x


class GRU_Base(Net):
    """
    The base class for GRU networks.
    """

    def __init__(self, in_dim, layers):
        super(GRU_Base, self).__init__()
        self.layers = create_layers(nn.GRUCell, in_dim, layers)

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]

    def _base_forward(self, x):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))

            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.layers):
                    h = self.hidden[idx]
                    self.hidden[idx] = layer(x_t, h)
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])
        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.layers):
                h = self.hidden[idx]
                self.hidden[idx] = layer(x, h)
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)
        return x


class CNN_Base(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=(3, 3),
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(3, 3),
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=(3, 3),
                      padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
            # nn.Conv2d(in_channels=64,
            #           out_channels=64,
            #           kernel_size=(3, 3),
            #           padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=4)
        )

        flat_features = self.compute_flat_feature(input_dim)

        self.linear_layer = nn.Linear(in_features=flat_features,
                                      out_features=output_dim)

    def compute_flat_feature(self, shape):
        out = self.conv_block(torch.rand(shape))
        num_features = np.prod(out.shape)
        return num_features

    def forward(self, x):
        if len(x.shape) == 3:
            x = self.conv_block(x)
            x = x.view(-1)
            x = self.linear_layer(x)
        elif len(x.shape) == 5:
            num_traj, num_batch = x.shape[:2]
            x = x.reshape(-1, *x.shape[-3:])
            x = self.conv_block(x)
            x = x.view(num_traj * num_batch, - 1)
            x = self.linear_layer(x)
            x = x.reshape(num_traj, num_batch, -1)
        else:
            raise Exception(f'dim of x must be 3 (single input) or 5 (batched trajectories)')

        return x
