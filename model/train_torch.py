import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from channelwise_GRU_torch_2 import *

embedding_dim = 128
hidden_units = 48
max_len = 10
epochs = 10
batch_size = 32

ob = Metesre(embedding_dim=embedding_dim, hidden_units=hidden_units, max_len=max_len)

ob.train(epoch=1)