import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    labels = torch.arange(batch_size)
    labels = torch.cat([labels, labels]).to(z1.device)

    mask = torch.eye(2 * batch_size).bool().to(z1.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    positives = torch.sum(z1 * z2, dim=-1)
    positives = torch.cat([positives, positives])

    numerator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=1)

    loss = -torch.log(numerator / denominator)
    return loss.mean()
