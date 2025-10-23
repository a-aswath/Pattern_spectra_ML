from lvq.GIALVQ import GIALVQ
from lvq.IAALVQ import IAALVQ
import torch
import glob
from utils.io_management import *
from utils.preprocessing import *
from utils.segmentation import build_informed_tree, cut_tree, segment_gialvq
from utils.visualization import *
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
# from sklearn_lvq import GrmlvqModel
# from sklearn_lvq import LgmlvqModel
import cv2
import numpy as np
# from sklearn_lvq.utils import plot2d
from collections import defaultdict
from skimage import exposure

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
def create_rgb_feature_vector(R_file, G_file, B_file, H_file, S_file, V_file):
    R = np.array(load_granulometry_m_file(R_file))[:10,:10].flatten()
    if np.max(R) == 0:
        return None
    
    # R= R/ np.max(R)
    # R= np.log1p(R)
    G = np.array(load_granulometry_m_file(G_file))[:10,:10].flatten()
    if np.max(G) == 0:
        return None
    
    # G= G/np.max(G)
    # G= np.log1p(G)
    B = np.array(load_granulometry_m_file(B_file))[:10,:10].flatten()
    if np.max(B) == 0:
        return None
    
    # B = B/np.max(B)
    # B= np.log1p(B)
    H = np.array(load_granulometry_m_file(H_file))[:10,:10].flatten()
    # H = np.log1p(H/np.max(H))
    # S=np.log1p(load_granulometry_m_file(S_file)[:10,:10].flatten())
    # S = np.log1p(S)
    # V=  np.log1p(load_granulometry_m_file(V_file)[:10,:10].flatten() )
    # V = np.log1p(V)
   
    # Make sure they are the same length
    if  len(R) != len(G) != len(B):
        raise ValueError("R, G, B arrays must be the same length")
    
    # Stack into N x 3 array
    rgb_features = np.concatenate([R, G, B, H], axis=0)
    return rgb_features

def build_gabor_kernel(ksize=55, sigma=4, theta=0, lambd=6, gamma=0.8, psi=0):
    return cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
def main():
    ds_name = "croptimal"
    s = '_mondial_55'# seed
    batch_sz = 5
    n_batches = 500

    avg_val_acc = 0.0
 

            
# Collect file lists (sorted so R,G,B match)
    R_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Mondial\Filtertype4\hR*.m")) #+sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\RudolphType1\scratch\p301644\Rudolph\xmaxtreetype1\hR*.m"))#+ sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\R2hM_*.m")) +sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\R2hR_*.m")) +sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\Spunta\R2h110_*.m"))
    G_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Mondial\Filtertype4\hG*.m")) #+sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\RudolphType1\scratch\p301644\Rudolph\xmaxtreetype1\hG*.m"))#+sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\G2hM_*.m"))+sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\G2hR_*.m"))+sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\Spunta\G2h110_*.m"))
    B_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Mondial\Filtertype4\hB*.m")) #+sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\RudolphType1\scratch\p301644\Rudolph\xmaxtreetype1\hB*.m"))#+ sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\B2hM_*.m"))+sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\B2hR_*.m"))+sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\Spunta\B2h110_*.m"))
    H_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Mondial\Filtertype4\hS*.m")) #+ sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\RudolphType1\scratch\p301644\Rudolph\xmaxtreetype1\hH*.m"))#+ sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\H2hM_*.m")) +sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\H2hR_*.m")) +sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\Spunta\H2h110_*.m"))
    import random


    random.seed(2)  # ← set a fixed seed (any number works)

    num_samples = 2000
    R_indices = random.sample(range(len(H_files)), num_samples)
    print(R_indices)
    # G_indices = random.sample(range(len(G_files)), num_samples)
    # B_indices = random.sample(range(len(B_files)), num_samples)
    R_files = [R_files[i] for i in R_indices]
    G_files = [G_files[i] for i in R_indices]
    B_files = [B_files[i] for i in R_indices]
    H_files = [H_files[i] for i in R_indices]#sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\H2hR_*.m"))
    S_files = None#sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\granulometrySpunta_G2h*.m"))
    V_files = None#sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\granulometrySpunta_B2h*.m"))

    num_bins = 121  # Because your feature vector has shape (121, 3)
    all_histogramsH = []#[] for _ in range(num_bins)]  # Pre-create 121 bins
    for i in range(len(B_files)):

        R_file = R_files[i]
        G_file = G_files[i]
        B_file = B_files[i] 
        H_file = H_files[i]
        # S_file = S_files[i]
        # V_file = V_files[i]
        rgb_feature = create_rgb_feature_vector(R_file, G_file, B_file, H_file, None, None)
            # Sort features into bins
        # for bin_idx in range(num_bins):
            # all_histogramsH[bin_idx].append(rgb_feature[bin_idx])
        if rgb_feature is not None:
            all_histogramsH.append(rgb_feature)
    # all_histogramsH = np.array(all_histogramsH)  # shape: (num_files, num_bins, 3)
    # Collect file lists (sorted so R,G,B match)
    # R_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\granulometrySpuntaRuhb*.m"))
    # G_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\granulometrySpuntaGuhb*.m"))
    # B_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\granulometrySpuntaBuhb*.m"))
    # H_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\granulometrySpuntaHuhb*.m"))
    # S_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\granulometrySpuntaSuhb*.m"))
    # V_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\granulometrySpuntaVuhb*.m"))
    # # num_bins = 20
    # all_histogramsnhb = []
    # num_bins = 121  # Because your feature vector has shape (121, 3)
    # # all_histogramsnH = [[] for _ in range(num_bins)]  # Pre-create 121 bins
    # for i in range(len(R_files)-1):

    #     R_file = R_files[i]
    #     G_file = G_files[i]
    #     B_file = B_files[i] 
    #     H_file = H_files[i]
    #     S_file = S_files[i] 
    #     V_file = V_files[i]
    #     rgb_feature = create_rgb_feature_vector(R_file, G_file, B_file, H_file, S_file, V_file)
    #     # for bin_idx in range(num_bins):
    #         # all_histogramsnH[bin_idx].append(rgb_feature[bin_idx])
    #     all_histogramsnhb.append(rgb_feature)
    # # all_histogramsnh = np.array(all_histogramsnh)  # shape: (num_files, 
    # ++9num_bins, 3)
    R_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Mondial\Filtertype4\uhR*.m"))#+ sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\RudolphType1\scratch\p301644\Rudolph\xmaxtreetype1\uhR*.m")) #+sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\R2uhR_*.m")) +sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\Spunta\R2uh110_*.m"))
    G_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Mondial\Filtertype4\uhG*.m"))#+ sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\RudolphType1\scratch\p301644\Rudolph\xmaxtreetype1\uhG*.m"))# +sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\G2uhR_*.m")) +sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\Spunta\G2uh110_*.m"))
    B_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Mondial\Filtertype4\uhB*.m"))#+ sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\RudolphType1\scratch\p301644\Rudolph\xmaxtreetype1\uhB*.m"))# +sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\B2uhR_*.m")) +sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\Spunta\B2uh110_*.m"))
    # R_files = random.sample(R_files, 400)  # randomly pick 10
    # G_files = random.sample(G_files, 400)
    # B_files = random.sample(B_files, 400)
    H_files = sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Mondial\Filtertype4\uhS*.m"))
    S_files = None#sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\G2huh*.m"))
    V_files = None#sorted(glob.glob(r"C:\Users\anush\Documents\PostDoc\Connected_operators\Michael\Xmaxtree\B2huh*.m"))
    # num_bins = 20
    all_histogramsnhy = []
    num_bins = 121  # Because your feature vector has shape (121, 3)
    # all_histogramsnH = [[] for _ in range(num_bins)]  # Pre-create 121 bin
    for i in range(len(H_files)):

        R_file = R_files[i]
        G_file = G_files[i]
        B_file = B_files[i] 
        H_file = H_files[i]
        # S_file = S_files[i] 
        # V_file = V_files[i]
        rgb_feature = create_rgb_feature_vector(R_file, G_file, B_file, H_file, None, None)
        # for bin_idx in range(num_bins):
            # all_histogramsnH[bin_idx].append(rgb_feature[bin_idx])
        if rgb_feature is not None:
            all_histogramsnhy.append(rgb_feature)
    # a1_flat = np.array(all_histogramsnhb).reshape(-1, 121*3)  # shape: (72*121, 3)
    import numpy as np
    a2_flat = np.array(all_histogramsnhy).reshape(-1, 100*4)
    b_flat = np.array(all_histogramsH).reshape(-1, 100*4)  # shape: (10*121, 3)
    # X1 = np.vstack([a1_flat, a2_flat])  # shape: (572*121 + 10*121, 3)
    # Combine into one dataset
    # X = np.vstack([all_histogramsnh, all_histogramsH])
    X1 = np.vstack([b_flat, a2_flat]) 
    # Create labels: 0 for nh, 1 for H
    y1 = np.array([0]*len(b_flat) + [1]*len(a2_flat)) #+ [2]*len(b_flat))
    batch_sz = 500
    n_batches = 1000
    # X1[:, np.all(X1 == 0, axis=0)] += 1e-12 
    from sklearn.model_selection import train_test_split


    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        X1, y1, test_size=.2, shuffle=True, random_state=42, stratify=y1
    )

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    import numpy as np

    # Example: 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Original indices
    # indices = np.arange(len(X1))

    # # Split data and indices together
    # x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
    #     X1, y1, indices,
    #     test_size=0.2,
    #     random_state=424,
    #     stratify=y1
    # )

    # Now you can see which original indices went where:
    # print("Train indices:", idx_train)
    # print("Test indices:", idx_test)
    # x_test, y_test = load_data(ds_name + str(s), test=False3)
    # x_train, y_train = load_data(ds_name + str(s), test=False)
    model = IAALVQ(max_iter=150, prototypes_per_class=20, omega_rank=10*4, seed=59,
                  regularization=0.0001
                  , omega_locality='PW', filter_bank=None,
                  block_eye=False, norm=False, correct_imbalance=True)
    print("Training...")
    # accuracies = []

    # for fold, (train_idx, test_idx) in enumerate(skf.split(X1, y1)):
    #     print(f" Fold {fold+1}")
        
    #     X_train, X_test = X1[train_idx], X1[test_idx]
    #     y_train, y_test = y1[train_idx], y1[test_idx]

    #     # Train your model here
    #     model.fit(X_train, y_train)
    #     show_cm(model, X_test, y_test, display_labels=['healthy', 'Yvirus'], title='Train')
    #     # Predict and evaluate
    #     y_pred = model.predict(X_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     accuracies.append(acc)
        
    #     print(f"Fold {fold+1} accuracy: {acc:.4f}")

    # # # ✅ Final cross-validation result
    # print(f"\nMean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    PARENT_DIR = r'C:\Users\anush\Documents\PostDoc\Croptimal datasets\mondial_norml\scratch\p301644\Mondial\xmaxtreed2'
    sys.path.append(PARENT_DIR)

    # mu = np.mean(x_train, axis=0)
    # stdi = np.std(x_train, axis=0)
    # x_train = x_train 
    
    import pickle

    # with open("ialvq_modelnegateSpunta.pkl", "rb") as f:
    #     model = pickle.load(f)
    # # # import pickle

    model.fit(x_train, y_train)
    # # # # # Assume `model` has been trained/fitted already
    # with open("ialvq_modelnegateSpunta.pkl", "wb") as f:
    #     pickle.dump(model, f)
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    f1_test= f1_score(y_test, y_pred_test, average='weighted')
    f1_train= f1_score(y_train, y_pred_train, average='weighted')
    print("Train F1-score: ", f1_train)
    print("Test F1-score: ", f1_test)
    print("Train accuracy: ", model.score(x_train, y_train))
    print("Test accuracy: ", model.score(x_test, y_test))

    # # confusion matrices
    show_cm(model, x_test, y_test, display_labels=['healthy', 'Yvirus'], title='Train')
    def normalize(omega):
        nf = np.sqrt(np.trace(omega.T.dot(omega)))
        omega = omega / nf
        return omega


    def lmbda(omega):
        return omega.T.dot(omega)


    def remove_diag(matrix):
        np.fill_diagonal(matrix, 0)
        return matrix

    # # feature correlation matrices
    show_lambdas(model.omegas_, class_names=['healthy', 'Yvirus'])
    def index_to_matrix_pos(flat_idx, matrix_shape=(10,10), n_matrices=3):
        rows, cols = matrix_shape
        matrix_size = rows * cols
        matrix_idx = flat_idx // matrix_size
        local_idx = flat_idx % matrix_size
        row = local_idx // cols
        col = local_idx % cols
        return matrix_idx, row, col
    def feature_importance_from_corr(lmbda_matrix):
    # ignore diagonal
        np.fill_diagonal(lmbda_matrix, 0)
        importance = np.mean(np.abs(lmbda_matrix), axis=0)  # average absolute correlation
        sorted_idx = np.argsort(importance)[::-1]   # sort descending
        return [(i, importance[i]) for i in sorted_idx]
    # features = feature_importance_from_corr(remove_diag(lmbda(model.omegas_[1])))
    diagonal= np.diag(lmbda(normalize(model.omegas_[1])))
    variances = np.array(diagonal)
    top_k = 20
    top_indices = np.argsort(variances)[-top_k:][::-1]

    for idx in top_indices:
        print(f"Feature {idx} importance: {variances[idx]}")
        
        matrix_idx, row, col = index_to_matrix_pos(idx)
        print(matrix_idx, row, col)  # 2, 10, 1  (0-based)
    # print(features+)
    show_prototypes(model.w_, 5, names=['healthy', 'Yvirus'], title='Prototypes')
    rank=model.omega_rank
    matrix = model.omegas_
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

    # # Example: top 6 eigenvalues + eigenvectors
    # top_vals, top_vecs, order = top_M_eigs(matrix, M=rank, by='abs')

    # print("Top 6 eigenvalues:")
    # print(top_vals)

    # print("\nCorresponding normalized eigenvectors (columns):")
    # print(top_vecs)
    # new_matrix = top_vecs * np.sqrt(top_vals)
    # X_reduced = x_train @ new_matrix
    # from sklearn.decomposition import PCA
    # X_scaled = X_reduced
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_scaled)
    # import matplotlib.pyplot as plt
    # print("Explained variance ratio:", pca.explained_variance_ratio_)
    # # Scatter plot of first 2 principal components
    # # Scatter plot with coloring by y
    # plt.figure(figsize=(8,6))
    # scatter = plt.scatter(     X_pca[:, 0], X_pca[:, 1],
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
    print("Segmentation (IALVQ only)...")
    img = skimage.io.imread(r'c:\Users\anush\Documents\PostDoc\Croptimal datasets\NAKFielddataset\Fontane\2025_6_3_Rietman fontanel _Fontane_Y-Virus_1.png')
    def color_balance_uint8(img, alpha):
        n_channels = img.shape[2]
        for ch in range(n_channels):
            histogram, bin_edges = np.histogram(img[..., ch], range=(0, 255), bins = 255)
            pmf = histogram / (img.shape[0] * img.shape[1])
            cmf = np.cumsum(pmf)
            idx = np.searchsorted(cmf, [alpha, 1 - alpha])
            left = 0
            if (pmf[idx[0]] < alpha):
                left = idx[0] + 1
            right = 255
            if (pmf[idx[1]] < alpha):
                right = idx[1]
            print(left)
            print(right)
            vals = img[..., ch].astype(np.float32)
            vals = np.round(255 * np.minimum(np.maximum((vals - left) / (right - left), 0), 1.0)).astype(np.uint8)
            img[..., ch] = vals
        return img
        # skimage loads in RGB → convert to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # gamma_corrected = exposure.adjust_gamma(img, .5)
    # cv2.imshow('gamma_corrected', gamma_corrected)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img = color_balance_uint8(img, .01)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # img = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
    if img.shape[-1] == 4:
        img = img[:, :, :3]  # Drop the alpha channel

  
    # skimage loads in RGB → convert to BGR for OpenCV
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert BGR to RGB (redundant, but follows your original step)
    # rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert BGR to HSV and cast to float for hue normalization
    # hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # image_lab = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2Lab)

    # Normalize hue to range [0, 1] based on 359° range (if that's your goal)
    # hsv[..., 0] *= 1.0 / 179.0
    # img_combined= np.concatenate((rgb, hsv), 2)
    # Standardization (Z-score normalization) on img_combined
    # mu = np.mean(img_combined, axis=(0, 1))  # Mean for each channel
    # std = np.std(img_combined, axis=(0, 1))  # Standard deviation for each channel

    # Apply the Z-score normalization
  

# Normalize image (similar to x_train_norm, if needed)
# Min-Max normalization to [0, 1]
    # # img_norm = (img_combined - img_combined.min()) / (img_combined.max() - img_combined.min() + 1e-8)
    # mu=np.zeros(9075)
    # stdi= np.ones(9075)

    # segment_gialvq(img, mu, stdi,model)

    #globalizing model:
    print("Charting...")
    model_global = GIALVQ(model)
    print("Train accuracy: ", model_global.score(x_train, y_train))
    print("Test accuracy: ", model_global.score(x_test, y_test))
    proj = model_global.project(x_train)
  
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(proj)
    import matplotlib.pyplot as plt
    print("Explained variance ratio:", pca.explained_variance_ratio_)
 
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_train,              # color by y (0 or 1)
        cmap="coolwarm",    # choose a colormap (blue/red)
        alpha=0.7
    )

    plt.xlabel("PC 1 (%.2f%%)" % (pca.explained_variance_ratio_[0] * 100))
    plt.ylabel("PC 2 (%.2f%%)" % (pca.explained_variance_ratio_[1] * 100))
    plt.title("PCA Projection Colored by Labels")
    plt.grid(True)
    plt.colorbar(scatter, label="Class")
    plt.show()


    # show_prototypes(model_global.w_global,11, names=['healthy',  'soil'], title='Prototypes')

    # grmlvq = GrmlvqModel()
    # grmlvq.fit(x_train, y_train)
    # print(grmlvq.lambda_)
  
    # plot2d(model_global, x_train, y_train, 1, 'grmlvq')

    # print('classification accuracy:', grmlvq.score(x_train, y_train))
    # plt.show()

    # segmentation

    # # img = Image.open(r'data\to_segment\Fontane_1_4e66dce5-5e9a-4107-b795-1c5790b2db72.png').convert("RGB")
    # # img = np.array(img)
    
    print("Segmentation...")
    atree, iqrs = build_informed_tree(model, x_train.reshape(-1,294), img_combined, patch_sz=7)#int(np.sqrt(model.omega_rank)))
    cut_tree(atree, iqrs, cls=1)

    print("Segmentation (GIALVQ only)...")
    mu=np.zeros(147)
    stdi= np.ones(147)
    segment_gialvq(img_combined, mu, stdi, model_global)
    print('done')

if __name__ == "__main__":
    main()
