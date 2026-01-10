import torch

dataset = "cifar10"
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")