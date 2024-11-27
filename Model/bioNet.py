# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
class BioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: 3x300x300, Output: 32x300x300
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # Output: 32x150x150
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: 64x150x150
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # Output: 64x75x75
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Output: 128x75x75
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # Output: 128x37x37

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Output: 256x37x37
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 256x18x18

            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Output: 512x18x18
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 512x9x9

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # Output: 1024x4x4
            
        )
        self.fc = nn.Linear( 1024*4*4, 512)

        self.metricNet = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.ReLU(),
            nn.Linear(512, 512//2),
            nn.ReLU(),
            nn.Linear(512//2, 512//4),
            nn.ReLU(),
            nn.Linear(512//4, 512//8),
            nn.ReLU(),
            nn.Linear(512//8, 512//16),
            nn.ReLU(),
            nn.Linear(512//16, 1)

        )
    
    def forward(self, img1, img2, batch_size):
        

        feature_vector1 = self.conv_layers(img1)
        feature_vector1 = self.fc(feature_vector1.view(feature_vector1.size(0), -1))

        feature_vector2 = self.conv_layers(img2)
        feature_vector2 = self.fc(feature_vector2.view(feature_vector2.size(0), -1))
        
        # Concatenate the features along the correct dimension
        feature_vector = torch.cat((feature_vector1, feature_vector2), dim=1)
        return self.metricNet(feature_vector)


class BioNetLoss(nn.Module):
    """
    Custom loss function that computes the loss based on the given embeddings.
    Suitable for neural networks.
    """
    def __init__(self, p, mu_n, mu_m, sigma):
        """
        Initialize the loss function.

        Parameters:
        - p: Dimensionality of the data (int).
        """
        super(BioNetLoss, self).__init__()
        self.p = p  # Dimensionality of the feature space
        self.mu_n = mu_n
        self.mu_m = mu_m
        self.sigma = sigma

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
        if z_batch.size(0) <= 1:
            sigma_batch = sigma
        
        else:
            sigma_batch = torch.var(z_batch, dim=0, unbiased=True) + 1e-6 #avoid division by 0
    
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
        loss = 0.5 * (log_det_term - self.p + trace_term + lst)

        return loss[0][0]
    def sort_labels(self, distances, labels):
    
        z_n = [z for z, y in zip(distances, labels) if y == 0]
        z_m = [z for z, y in zip(distances, labels) if y == 1]

        z_n = torch.stack(z_n, dim=0) if z_n else torch.empty(0, dtype=torch.float32)
        z_m = torch.stack(z_m, dim=0) if z_m else torch.empty(0, dtype=torch.float32)

        return z_n, z_m
    def threshold(self):
        
        phi = (self.mu_n + self.mu_m) / 2

        return phi
    def forward(self, distances, labels):
        """
        Forward pass to compute the total loss.

        Parameters:

        Returns:
        - Total loss value (torch.Tensor).
        """
        
        z_sn, z_sm = self.sort_labels(distances, labels)
        loss_n = torch.tensor(0)
        loss_m = torch.tensor(0)
        if z_sn.shape[0] > 0:
            loss_n = self.compute_loss_term(self.mu_n, z_sn, self.sigma)
        if z_sm.shape[0] > 0:
            loss_m = self.compute_loss_term(self.mu_m, z_sm, self.sigma)

        return loss_n + loss_m


