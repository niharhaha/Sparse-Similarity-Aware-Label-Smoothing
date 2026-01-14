import torch
import torch.nn.functional as F

def histogram_ece(logits, true_labels, num_bins=15):
    probs = F.softmax(logits, dim=1)
    N, K = probs.shape

    labels_expanded = true_labels.unsqueeze(1).expand_as(probs)
    classes = torch.arange(K, device=logits.device).unsqueeze(0).expand_as(probs)

    correct_matrix = (labels_expanded == classes).float()

    probs_flat = probs.flatten()
    correct_flat = correct_matrix.flatten()

    bin_edges = torch.linspace(0, 1.0, num_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)

    for m in range(num_bins):
        start = bin_edges[m]
        end = bin_edges[m + 1]

        in_bin = (probs_flat > start) & (probs_flat <= end)
        num_in_bin = in_bin.sum()

        if num_in_bin > 0:
            lik_in_bin = correct_flat[in_bin].mean()
            conf_in_bin = probs_flat[in_bin].mean()

            ece += (num_in_bin / (N * K)) * torch.abs(lik_in_bin - conf_in_bin)

    return ece.item()

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
