
import torch
import torch.nn as nn
import numpy as np
import math

class gmlvq_net(nn.Module):
    def __init__(self, n_feats, initial_protos, proto_labels, use_matrix_per_proto, mu, std):
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
        
        self.proto_labels = torch.tensor(proto_labels)

    def matrix(self):
        if self.use_matrix_per_proto:
            mat = self.mat
        else:
            mat = self.mat            

        # Ensure positive semi-definite
        return mat @ mat.T 
    
    def proto_distances(self, x):
        x = (x - self.mu) * self.stdi

        distances = []
        for i in range(self.protos.shape[0]):
            mat = self.matrix()
            diff = self.protos[i] - x
            result = (diff * (diff @ mat.T)).sum(axis = 1)
            distances.append(result)
        distances = torch.stack(distances, axis = 1)            
        return distances

    def forward(self, x):
        with torch.no_grad():
            distances = self.proto_distances(x)
            return self.proto_labels[distances.argmin(1)]

class gmlvq():
    LR = 0.01

    def optimizer(self):
        return torch.optim.AdamW(self.net.parameters(), lr = self.LR, weight_decay = 0)


    def initialize(self, n_feats, initial_protos, proto_labels, use_matrix_per_proto, mu, std):
        self.net = gmlvq_net(n_feats, initial_protos, proto_labels, use_matrix_per_proto, mu, std)
        self.optimizer = self.optimizer()

    def save(self, file):
        torch.save(self.net, file)

    def load(self, file):
        self.net = torch.load(file, weights_only=False)
        self.optimizer = self.optimizer()


    def loss_fn(self, x, labels):
        # distances between features and prototypes
        distances = self.net.proto_distances(x)
        # correct classes
        correct = self.net.proto_labels.unsqueeze(0).eq(labels.unsqueeze(1))
        # minimum distance to a prototype with the same label
        min_distance_same = distances.maximum((~correct) * np.finfo(np.float32).max).min(1).values
        # minimum distance to a prototype with a different label
        min_distance_other = distances.maximum((correct) * np.finfo(np.float32).max).min(1).values
        loss = ((min_distance_same - min_distance_other) / (min_distance_same + min_distance_other)).mean()

        return loss
    
    def train(self, x, labels):
        self.net.train()
        self.optimizer.zero_grad()    
        loss = self.loss_fn(x, labels)
        loss.backward()        
        self.optimizer.step()
        
        with torch.no_grad():
            if self.net.use_matrix_per_proto:
                for i in range(self.net.protos.shape[0]):
                    self.net.mat[i] /= self.net.mat[i].norm()
            else:
                self.net.mat /= self.net.mat.norm()
            
        return loss
    
    def eval(self, x):
        self.net.eval()
        return self.net(x)