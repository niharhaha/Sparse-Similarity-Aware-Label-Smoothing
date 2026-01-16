import torch
import torch.nn.functional as F

def top_label_ece(logits, labels, n_bins=15):
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)

    correctness = (predictions == labels).float()
    bin_edges = torch.linspace(0, 1.0, n_bins + 1, device=logits.device)

    ece = torch.zeros(1, device=logits.device)

    for m in range(n_bins):
        start = bin_edges[m]
        end = bin_edges[m + 1]

        in_bin = (confidences > start) & (confidences <= end)
        num_in_bin = in_bin.sum()

        if num_in_bin > 0:
            acc_in_bin = correctness[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += (num_in_bin.float() / len(confidences)) * torch.abs(acc_in_bin - conf_in_bin)

    return ece.item()

