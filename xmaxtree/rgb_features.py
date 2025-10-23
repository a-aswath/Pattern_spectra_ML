import numpy as np
import glob
import scipy.io as sio
def load_granulometry_m_file(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[2:]  # skip header + column indices

    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        numbers = [float(x) for x in tokens[1:]]  # skip row index
        data.append(numbers)
    return np.array(data)
def normalize_channel(arr, method="minmax"):
    if method == "minmax":
        # Avoid divide by zero
        min_val, max_val = np.min(arr), np.max(arr)
        return (arr - min_val) / (max_val - min_val + 1e-8)
    elif method == "zscore":
        mean, std = np.mean(arr), np.std(arr)
        return (arr - mean) / (std + 1e-8)
    else:
        raise ValueError("Unknown normalization method")
def create_rgb_feature_vector(R_file, G_file, B_file, H_file, S_file, V_file):
    # R = load_granulometry_m_file(R_file).flatten() 
    # G = load_granulometry_m_file(G_file).flatten() 
    # B = load_granulometry_m_file(B_file).flatten() 
    H = load_granulometry_m_file(H_file).flatten()      
    S = load_granulometry_m_file(S_file).flatten()     
    V = load_granulometry_m_file(V_file).flatten()     
    # Make sure they are the same length
    if  len(H) != len(S) != len(V):
        raise ValueError("R, G, B arrays must be the same length")
    
    # Stack into N x 3 array
    rgb_features = np.log(np.concatenate([ H, S, V], axis=0) + 1e-8)
    return rgb_features
# Collect file lists (sorted so R,G,B match)
R_files = sorted(glob.glob("granulometryRh*.m"))
G_files = sorted(glob.glob("granulometryGh*.m"))
B_files = sorted(glob.glob("granulometryBh*.m"))
H_files = sorted(glob.glob("granulometry20Hh*.m"))
S_files = sorted(glob.glob("granulometry20Sh*.m"))
V_files = sorted(glob.glob("granulometry20Vh*.m"))

num_bins = 121  # Because your feature vector has shape (121, 3)
all_histogramsH = []#[] for _ in range(num_bins)]  # Pre-create 121 bins
for i in range(len(H_files)-1):

    R_file = R_files[i]
    G_file = G_files[i]
    B_file = B_files[i] 
    H_file = H_files[i]
    S_file = S_files[i]
    V_file = V_files[i]
    rgb_feature = create_rgb_feature_vector(R_file, G_file, B_file, H_file, S_file, V_file)
        # Sort features into bins
    # for bin_idx in range(num_bins):
        # all_histogramsH[bin_idx].append(rgb_feature[bin_idx])
    all_histogramsH.append(rgb_feature)
# all_histogramsH = np.array(all_histogramsH)  # shape: (num_files, num_bins, 3)
# Collect file lists (sorted so R,G,B match)
R_files = sorted(glob.glob("granulometryRuh*.m"))
G_files = sorted(glob.glob("granulometryGuh*.m"))
B_files = sorted(glob.glob("granulometryBuh*.m"))
H_files = sorted(glob.glob("granulometry20Huh*.m"))
S_files = sorted(glob.glob("granulometry20Suh*.m"))
V_files = sorted(glob.glob("granulometry20Vuh*.m"))
# num_bins = 20
all_histogramsnh = []
num_bins = 121  # Because your feature vector has shape (121, 3)
# all_histogramsnH = [[] for _ in range(num_bins)]  # Pre-create 121 bins
for i in range(len(H_files)-1):

    R_file = R_files[i]
    G_file = G_files[i]
    B_file = B_files[i] 
    H_file = H_files[i]
    S_file = S_files[i] 
    V_file = V_files[i]
    rgb_feature = create_rgb_feature_vector(R_file, G_file, B_file, H_file, S_file, V_file)
    # for bin_idx in range(num_bins):
        # all_histogramsnH[bin_idx].append(rgb_feature[bin_idx])
    all_histogramsnh.append(rgb_feature)
# all_histogramsnh = np.array(all_histogramsnh)  # shape: (num_files, num_bins, 3)

a_flat = np.array(all_histogramsnh).reshape(-1, 441*3)  # shape: (72*121, 3)
b_flat = np.array(all_histogramsH).reshape(-1, 441*3)  # shape: (10*121, 3)
X1 = np.vstack([a_flat, b_flat])  # shape: (72*121 + 10*121, 3)
# Combine into one dataset
# X = np.vstack([all_histogramsnh, all_histogramsH])

# Create labels: 0 for nh, 1 for H
y1 = np.array([0]*len(a_flat) + [1]*len(b_flat))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X1)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
import matplotlib.pyplot as plt
print("Explained variance ratio:", pca.explained_variance_ratio_)
# Scatter plot of first 2 principal components
# Scatter plot with coloring by y
# plt.figure(figsize=(8,6))
# scatter = plt.scatter(
#     X_pca[:, 0], X_pca[:, 1],
#     c=y1,                # color by y (0 or 1)
#     cmap="coolwarm",    # choose a colormap (blue/red)
#     alpha=0.7
# )

# plt.xlabel("PC 1 (%.2f%%)" % (pca.explained_variance_ratio_[0] * 100))
# plt.ylabel("PC 2 (%.2f%%)" % (pca.explained_variance_ratio_[1] * 100))
# plt.title("PCA Projection Colored by Labels")
# plt.grid(True)
# plt.colorbar(scatter, label="Class")
# plt.show()
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

clf = LogisticRegression()
scores = cross_val_score(clf, X_scaled, y1, cv=5)
print("5-fold CV accuracy:", scores.mean())

def xval(imgs):
    imgs_train = []
    imgs_val = []
    n_classes = 2
    
    for i in range(n_classes):    
        val_idx = np.random.randint(0, len(imgs[i]))
        val = [imgs[i][val_idx]]
        train = imgs[i].copy()
        del train[val_idx]
        imgs_train.append(train)
        imgs_val.append(val)
    
    return imgs_train, imgs_val
def feats_mean_std(imgs, batch_sz, n_batches):
    stats_mu = []
    stats_M2 = []
    X_train_tensor = torch.from_numpy(imgs).float()
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # train_loader = data_loader(imgs, batch_sz)
    # train_iter = 
    for batch_nr in range(n_batches):
        X = next(iter(train_loader))
       
 
      
        M2, mu = torch.var_mean(X[0], axis = 0, correction = 0)
        M2 *= batch_sz
        
        stats_mu.append(mu)
        stats_M2.append(M2)

    mu = torch.zeros(stats_mu[0].shape[0])
    M2 = torch.zeros(stats_M2[0].shape[0])
    
    for i in range(len(stats_mu)):
        delta = stats_mu[i] - mu        
        mu += delta / (i + 1.0)
        M2 += stats_M2[i] + delta * delta * i / (i + 1.0)
            
    return mu, torch.sqrt(M2 / (batch_sz * n_batches - 1))        
    
from gmlvq import gmlvq
import torch

batch_sz = 500
n_batches = 1000
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
avg_val_acc = 0.0
# Split X and y together
imgs_train, imgs_val, y_train, y_val = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1  # stratify=y keeps class balance
)
# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(imgs_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_val_tensor   = torch.from_numpy(imgs_val).float()
y_val_tensor   = torch.from_numpy(y_val).long()

# # Wrap into TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)

# DataLoaders
# batch_sz = 64
train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)

mu, std = feats_mean_std(imgs_train, batch_sz, n_batches // 10)
n_feats = X1.shape[1]
X_0 = X1[y1 == 0, :]
X_1 = X1[y1 == 1, :]
init_protos = torch.from_numpy(np.array([X_0.mean(0), X_1.mean(0)])).float()

lvq = gmlvq()
lvq.initialize(n_feats, init_protos, [0, 1], False, mu, std)
# for data_split_nr in range(5):
#     print(f'data split #{data_split_nr}')


    
#     # train_loader = DataLoader(imgs_train, batch_sz)
#     X, y = next(iter(train_loader))

    
for batch_nr in range(n_batches):
    X, y = next(iter(train_loader))
    # X = torch(X)
    # y = torch(y)        
    lvq.train(X, y)
        

val_acc = 0.0
n_samples = 0

for X, y in val_loader:
    y_pred = lvq.eval(X)  # assuming this returns predicted labels
    
    correct = (y_pred == y).sum().item()
    val_acc += correct
    n_samples += y.shape[0]

avg_val_acc = val_acc / n_samples

print(f'avg val acc: {avg_val_acc}')
matrix=lvq.net.matrix().detach().cpu().numpy()

def top_M_eigs(M_tensor, M=5, tol=1e-20, by='abs'):
    # detach and convert to float64 for numerical stability
    M_tensor = M_tensor#.detach().cpu().to(torch.float64)

    # symmetric eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(M_tensor)

    # convert to numpy
    eigvals = eigvals
    eigvecs = eigvecs

    # filter non-zero eigenvalues
    nonzero_mask = abs(eigvals) > tol
    eigvals = eigvals[nonzero_mask]
    eigvecs = eigvecs[:, nonzero_mask]

    # order indices
    if by == 'abs':
        order = np.argsort(-np.abs(eigvals))
    elif by == 'real':
        order = np.argsort(-np.real(eigvals))
    else:
        order = np.argsort(-eigvals)

    # reorder eigenvalues and eigenvectors
    eigvals = eigvals[order][:min(M, eigvals.size)]
    eigvecs = eigvecs[:, order][:, :min(M, eigvals.size)]

    return eigvals, eigvecs, order

# Example: top 6 eigenvalues + eigenvectors
top_vals, top_vecs, order = top_M_eigs(matrix, M=6, by='abs')

print("Top 6 eigenvalues:")
print(top_vals)

print("\nCorresponding normalized eigenvectors (columns):")
print(top_vecs)
new_matrix = top_vecs * np.sqrt(top_vals)
X_reduced = X_scaled @ new_matrix
from sklearn.decomposition import PCA
X_scaled = X_reduced
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
import matplotlib.pyplot as plt
print("Explained variance ratio:", pca.explained_variance_ratio_)
# Scatter plot of first 2 principal components
# Scatter plot with coloring by y
plt.figure(figsize=(8,6))
scatter = plt.scatter(     X_pca[:, 0], X_pca[:, 1],
     c=y1,                # color by y (0 or 1)
     cmap="coolwarm",    # choose a colormap (blue/red)
     alpha=0.7
 )

plt.xlabel("PC 1 (%.2f%%)" % (pca.explained_variance_ratio_[0] * 100))
plt.ylabel("PC 2 (%.2f%%)" % (pca.explained_variance_ratio_[1] * 100))
plt.title("PCA Projection Colored by Labels")
plt.grid(True)
plt.colorbar(scatter, label="Class")
plt.show()
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], label='nh', alpha=0.6)
plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], label='H', alpha=0.6)
plt.xlabel('t-SNE dim 1')
plt.ylabel('t-SNE dim 2')
plt.legend()
plt.title('t-SNE of your features')
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='nh', alpha=0.6)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='H', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('PCA of your features')
plt.show()