# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

from featureNet import FeatureNet
from metricNet import MetricNet

# %%
class BioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.featureNet = FeatureNet()
        self.metricNet = MetricNet()
    
    def forward(self, x):
        # (2, 3, 299, 299) -> (1, 1024)
        feature_vector = torch.tensor([])
        for image in x:
            image = image.unsqueeze(0)
            print(image.shape)
            image_tensor = self.featureNet(image)
            feature_vector = torch.cat(tensors=(feature_vector, image_tensor), dim=1)
        print(feature_vector.shape)
        return self.metricNet(feature_vector)[0]




class BioNetLoss(nn.Module):
    """
    Custom loss function that computes the loss based on the given embeddings.
    Suitable for neural networks.
    """
    def __init__(self, p):
        """
        Initialize the loss function.

        Parameters:
        - p: Dimensionality of the data (int).
        """
        super(BioNetLoss, self).__init__()
        self.p = p  # Dimensionality of the feature space

    def compute_loss_term(self, mu, z_batch, sigma):
        """
        Compute a single loss term for the given total and batch embeddings.

        Parameters:
        - z_tot: Tensor of total embeddings (torch.Tensor).
        - z_batch: Tensor of batch embeddings (torch.Tensor).

        Returns:
        - Loss term value (torch.Tensor).
        """
        # Compute covariance matrice of the samples
        sigma_batch = torch.var(z_batch, dim=0, unbiased=True)
        
        # Compute determinants and inverse
        det_sigma_s = sigma_batch
        inv_sigma = 1/sigma

        # Log determinant, first term
        log_det_term = torch.log(sigma / det_sigma_s)

        # Trace term, second term
        trace_term = inv_sigma * sigma_batch

        # Last term
        diff = mu - torch.mean(z_batch, dim=0)
        lst = diff * inv_sigma * diff  

        return 0.5 * (log_det_term - self.p + trace_term + lst)

    def forward(self, mu_n, z_sn, mu_m, z_sm, sigma):
        """
        Forward pass to compute the total loss.

        Parameters:
        - z_n: Tensor of normal samples (torch.Tensor).
        - z_sn: Tensor of normal batch samples (torch.Tensor).
        - z_m: Tensor of abnormal samples (torch.Tensor).
        - z_sm: Tensor of abnormal batch samples (torch.Tensor).

        Returns:
        - Total loss value (torch.Tensor).
        """
        loss_n = self.compute_loss_term(mu_n, z_sn, sigma)
        loss_m = self.compute_loss_term(mu_m, z_sm, sigma)
        return loss_n + loss_m


def threshold(mu_n, mu_m):
    """
    Compute the threshold value for the given embeddings using PyTorch.

    Parameters:
    - z_n, z_m: Tensors of embeddings (torch.Tensor).

    Returns:
    - Threshold value (torch.Tensor).
    """
    return (mu_n + mu_m) / 2


