import torch 
import torch.nn as nn 

class RMSNorm(nn.Module):
    def __init__(self, model_dimension:int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        # learnable scale parameter
        self.weight = nn.Parameter(torch.ones(model_dimension))
    
    def forward(self, x:torch.Tensor):
        # mean of squares along last dimension
        mean_squares = x.pow(2).mean(dim=-1, keepdim=True)
        # Root Mean Squre
        root_mean_square = torch.sqrt(mean_squares + self.epsilon)
        # normalize
        x_normalize = x / root_mean_square
        # scale
        return x_normalize * self.weight