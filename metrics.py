import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, wilcoxon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def top_label_ece(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return ece.item()

def max_calibration_error(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece = torch.max(torch.abs(acc_in_bin - conf_in_bin), ece)

    return ece.item()

def nll_loss(logits, labels):
    return F.cross_entropy(logits, labels, reduction="mean").item()

def accuracy(model, loader, k = (1, 5)):
    model.eval()
    correct = {key:0 for key in k}
    total = 0

    maxk = max(k)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)

            for key in k:
                correct[key] += (pred[:, :key] == y.view(-1, 1)).any(dim=1).sum().item()
            total += y.size(0)

    return {key: correct[key] / total for key in k}

