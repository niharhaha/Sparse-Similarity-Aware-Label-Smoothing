import torch

dataset = "cifar100"
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")