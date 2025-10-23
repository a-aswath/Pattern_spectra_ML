import torch
import torch.nn as nn
import numpy as np
import math

class alvq_net(nn.Module):
    def __init__(self, n_feats, initial_protos, use_matrix_per_proto, mu, std):
        super().__init__()
        self.protos = nn.Parameter(initial_protos.clone(), requires_grad=True)        
        self.use_matrix_per_proto = use_matrix_per_proto
        self.mu = nn.Parameter(mu.clone(), requires_grad=False)
        stdi = 1.0 / std.clone()
        stdi[std == 0.0] = 0
        self.stdi = nn.Parameter(stdi, requires_grad=False)
        mat = torch.eye(n_feats, requires_grad=False) / np.sqrt(n_feats)
        if use_matrix_per_proto:
            mat = mat.repeat(initial_protos.shape[0], 1, 1)
        self.mat = nn.Parameter(mat, requires_grad=True)
        
    def matrix(self, proto_nr):
        if self.use_matrix_per_proto:
            mat = self.mat[proto_nr]
        else:
            mat = self.mat            

        # Ensure positive semi-definite
        return mat.T @ mat       

    def length_in_space(self, mat, x):
        n_feats = x.shape[-1]
        mapped_x = (mat @ x.reshape(-1, n_feats, 1)).reshape(1, -1, n_feats)
        x = x.reshape(1, -1, n_feats) * mapped_x
        return torch.sqrt(x.sum(2)).squeeze()

    def probabilities(self, x):
        x = self.angular_dissimilarities(x)
        beta = 1
        x = (torch.exp(-beta * (x - 1)) - 1) / (np.exp(beta * 2) - 1)
        x = x / x.sum(1, keepdim=True)     
        return x         
    
    def angular_dissimilarities(self, x):
        n_feats = x.shape[-1]
        
        if (x.dim() > 2):
            x = x.reshape(-1, n_feats)        

        x = (x - self.mu) * self.stdi

        similarities = []
        for i in range(self.protos.shape[0]):
            mat = self.matrix(i)
            x_lens = self.length_in_space(mat, x)
            proto_len = self.length_in_space(mat, self.protos[i])
            lens = x_lens * proto_len + 1e-10
            mapped_proto = (mat @ self.protos[i].reshape(n_feats, 1)).reshape(1, n_feats)
            z = x * mapped_proto
            sim = z.sum(1) / lens
            similarities.append(sim)
            
        similarities = torch.stack(similarities, axis = 1)                    
        
        return similarities        

    def post_optimizer(self):
        # normalize matrices
        with torch.no_grad():
            if self.use_matrix_per_proto:
                for i in range(self.protos.shape[0]):
                    self.mat[i] /= torch.sqrt(self.matrix(i).trace())
            else:
                self.mat /= torch.sqrt(self.matrix(0).trace())

    def inference(self, x):
        with torch.no_grad():
            return self.angular_dissimilarities(x).argmin(1)