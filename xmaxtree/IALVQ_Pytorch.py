import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class IALVQ_Pytorch(nn.Module):
    def __init__(self,  n_classes, n_features, prototypes_per_class=10, omega_rank=None, 
                 initial_prototypes=None, initial_omegas=None, omega_locality='PW', 
                 block_eye=False, norm=True, random_state=None, class_weights=None, seed=None):
        super(IALVQ_Pytorch, self).__init__()
        
        # Set up parameters
 
        # Set up parameters
        self.n_classes = n_classes
        self.n_features = n_features
        self.nb_features = n_features  # Set nb_features explicitly to n_features
        self.prototypes_per_class = prototypes_per_class
        self.omega_rank = omega_rank if omega_rank is not None else n_features
        self.initial_prototypes = initial_prototypes
        self.initial_omegas = initial_omegas
        self.omega_locality = omega_locality
        self.block_eye = block_eye
        self.norm = norm
        self.random_state = np.random.RandomState(seed)  # or torch.manual_seed(seed)
        self.class_weights = class_weights
        
        # Initialize parameters that depend on each other
        self.nb_classes = self.n_classes  # Now explicitly set the number of classes
        self.nb_omegas = self.nb_classes  # Define the number of omegas based on classes
        
        # Initialize the prototypes and omegas
        # self._set_prototypes()
        self._set_omegas()
        
        # Initialize the model with training data and labels
        self.samples = None  # Placeholder for training data
        self.labels = None   # Placeholder for training labels
        


    def _initialize(self, x, y):
        """
        Initialize model with the training data.
        :param x: train samples
        :param y: train labels
        """
        self.samples = np.array(x)
        self.labels = np.array(y)
        self.nb_samples, self.nb_features = self.samples.shape
        self.classes_ = np.unique(self.labels)  # Use np.unique to get unique class labels
        self.nb_classes = len(self.classes_)
        
        # Set prototypes and omegas
        self._set_prototypes()
        self._set_omegas()
    def _set_prototypes(self):
        """
        Set the prototype vectors, either by initializing them as means with noise 
        or using user-provided initial prototypes.
        """
        nb_ppc = np.ones([self.nb_classes], dtype='int') * self.prototypes_per_class
        if self.initial_prototypes is None:  # init as means with noise
            # self.w_ = np.empty([np.sum(nb_ppc), self.nb_features], dtype=np.double)
            self.c_w_ = torch.randn(self.nb_classes, self.n_features) 
            self.w_ = np.ones((self.c_w_.shape[0], self.c_w_.shape[0]))  # Use self.c_w_.shape[0] to get the correct dimension
            self.w_ = torch.tensor(self.w_, dtype=torch.float32)  # Convert to tensor
            pos = 0
            for cls in range(self.nb_classes):
                nb_prot = nb_ppc[cls]
                mean = np.mean(self.samples[self.labels == self.classes_[cls], :], axis=0)
                # Ensure the right-hand side is a PyTorch tensor
                self.w_[pos:pos + nb_prot] = torch.tensor(mean + (self.random_state.rand(nb_prot, self.nb_features) * 2 - 1), dtype=torch.float32)

                self.c_w_[pos:pos + nb_prot] = self.classes_[cls]
                pos += nb_prot
        else:
            self.w_ = self.initial_prototypes[:, :-1]
            self.c_w_ = self.initial_prototypes[:, -1].astype(int)

        # Convert prototypes into nn.Parameter
        self.w_ = nn.Parameter(torch.tensor(self.w_, dtype=torch.float32), requires_grad=True)  # Now trainable
        self.c_w_ = torch.tensor(self.c_w_, dtype=torch.int64)  # Keep labels as non-trainable

        self.nb_prototypes = self.c_w_.shape[0]
        self.c_ = np.ones((self.c_w_.size, self.c_w_.size))

    def _set_omegas(self):
  
        self.nb_omegas = self.nb_classes  # Using one omega per class as per your setup
        self.split_indices = np.arange(self.omega_rank, self.omega_rank * self.nb_omegas, self.omega_rank)

        if self.omega_rank is None:
            self.omega_rank = self.n_features

        if self.initial_omegas is None:
            omegas_list = []  # Initialize a list to store omegas
            for omega in range(self.nb_omegas):
                if self.block_eye:
                    # Identity matrix with reshaping
                    eye = np.eye(self.n_features // self.channel_num, self.n_features // self.channel_num)
                    omega = np.stack([eye for _ in range(self.channel_num)]).ravel().reshape(self.n_features, 
                                                                                            self.n_features // self.channel_num)
                    omega = omega.T
                    if self.norm:
                        nf = np.sqrt(np.trace(omega.T.dot(omega)))
                        omega = omega / nf
                    omegas_list.append(omega)
                else:
                    # Randomly initialized omega matrix
                    omega = self.random_state.rand(self.omega_rank, self.n_features) * 2.0 - 1.0
                    if self.norm:
                        nf = np.sqrt(np.trace(omega.T.dot(omega)))
                        omega = omega / nf
                    omegas_list.append(omega)

            # Convert the list of omegas to a tensor
            omegas_tensor = torch.tensor(omegas_list, dtype=torch.float32)

            # Wrap in nn.Parameter to make it learnable
            self.omegas_ = nn.Parameter(omegas_tensor, requires_grad=True)
        else:
            if not isinstance(self.initial_omegas, list):
                raise ValueError("Initial omegas must be a list")
            # Use the initial omegas if provided
            omegas_tensor = torch.tensor(self.initial_omegas, dtype=torch.float32)
            self.omegas_ = nn.Parameter(omegas_tensor, requires_grad=True)


    def forward(self, x,y):
        """
        Forward pass: Compute predictions based on input features x.
        :param x: input tensor
        :return: predictions
        """

                # Dynamically set training data if not already set
        if self.samples is None or self.labels is None:
            self.samples = x
            self.labels = y
            self.nb_samples, self.nb_features = self.samples.shape
            self.classes_ = np.unique(self.labels)
            self.nb_classes = len(self.classes_)
            self._set_prototypes()
        # Placeholder forward pass (modify to match your model logic)
        distances = self.compute_distances(x)
        preds = self.c_w_[torch.argmin(distances, dim=1)]
        return preds
    
    def compute_distances(self, x):
        """
        Compute distances between input x and prototypes.
        :param x: input features
        :return: distance tensor
        """
        distances = []
        for i, proto in enumerate(self.w_):
            dist = torch.norm(x - torch.tensor(proto), dim=1)  # Example distance computation
            distances.append(dist)
        return torch.stack(distances, dim=1)

    def loss_fn(self, x, y):
        distances = self.compute_distances(x)
        winner_idx = torch.argmin(distances, dim=1)
        winner_labels = self.prototype_labels[winner_idx]

        # Cost-sensitive loss
        if self.cost_matrix is not None:
            costs = self.cost_matrix[y, winner_labels]
            classification_loss = torch.mean(costs)
        else:
            loss_vals = (winner_labels != y).float()
            if self.class_weights is not None:
                weights = self.class_weights[y]
                loss_vals *= weights
            classification_loss = torch.mean(loss_vals)

        # Regularization: Encourage full-rank omegas (maximize logdet)
        reg_loss = 0.0
        if self.regularization > 0.0:
            for omega in self.omegas:
                gram = torch.matmul(omega, omega.t()) + 1e-6 * torch.eye(omega.shape[0], device=omega.device)
                reg_loss += -torch.logdet(gram)

        total_loss = classification_loss + self.regularization * reg_loss
        return total_loss

    def fit(self, x_train, y_train, max_iter=100, lr=1e-3):
        self._initialize(x_train, y_train)
        optimizer =optim.Adam(self.parameters(), lr=lr)
            # Ensure x_train and y_train are part of the computation graph
        # self._set_prototypes()
        x_train= x_train.detach()
        y_train= y_train.detach()
        for epoch in range(max_iter):
            optimizer.zero_grad()  # Zero the gradients before the forward pass

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self(x_train, y_train)

            # Compute the loss (you can define your custom loss function here)
            loss = self.loss_fn(x_train, y_train)
            # x_train.requires_grad = True  # Ensure x_train is part of the computation graph
            # Backward pass: Compute gradient of the loss with respect to model parameters
            loss.backward()

            # Optimize the model parameters using Adam
            optimizer.step()

            # Optional normalization of omega
            if self.norm_omegas:
                with torch.no_grad():
                    for i, omega in enumerate(self.omegas):
                        trace = torch.trace(omega @ omega.t())
                        if trace > 0:
                            self.omegas[i].div_(torch.sqrt(trace))

            # Log the loss every 10 epochs or at the last epoch
            if epoch % 10 == 0 or epoch == max_iter - 1:
                print(f"Epoch {epoch}/{max_iter}, Loss: {loss.item()}")


    def predict(self, x):
        return self.forward(x)

    def predict_proba(self, x):
        distances = self.compute_distances(x)
        logits = -distances  # smaller distances -> higher logit
        probs = torch.softmax(logits, dim=1)
        return probs
