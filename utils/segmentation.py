import matplotlib
import matplotlib.pyplot as plt

from utils.visualization import *

from scipy.special import softmax

from alpha_trees.AlphaTree import AlphaTree
import cv2
def softmin(distances, sigma):
    return softmax(-distances / sigma ** 2, 1)
import numpy as np

def predict_in_batches(image, model, mu, stdi, p=32, batch_size=1024):
    H, W, C = image.shape
    out_H = H - p + 1
    out_W = W - p + 1

    # Prepare output placeholders
    labels_out = np.zeros((out_H, out_W), dtype=np.int32)
    distances_out = np.zeros((out_H, out_W, 2), dtype=np.float32)

    idx = 0
    batch_patches = []

    coords = []
    for i in range(out_H):
        for j in range(out_W):
            patch = image[i:i+p, j:j+p, :].reshape(-1)  # Shape: (p*p*C,)
            patch = (patch - mu) / stdi
            batch_patches.append(patch)
            coords.append((i, j))
            idx += 1

            if idx % batch_size == 0 or (i == out_H - 1 and j == out_W - 1):
                batch_patches_np = np.array(batch_patches)
                labels, distances = model.predict(batch_patches_np, ret_dist=True)

                for (ii, jj), label, dist in zip(coords, labels, distances):
                    labels_out[ii, jj] = label
                    distances_out[ii, jj] = dist

                # Reset for next batch
                batch_patches = []
                coords = []

    return labels_out, distances_out
def predict_half_overlapping(image, model, mu, stdi, p=25, stride=None, batch_size=1024):
    if stride is None:
        stride = p // 4  # Half-overlapping

    H, W, C = image.shape
    out_H = (H - p) // stride + 1
    out_W = (W - p) // stride + 1

    labels_out = np.zeros((out_H, out_W), dtype=np.int32)
    distances_out = np.zeros((out_H, out_W, 2), dtype=np.float32)

    idx = 0
    batch_patches = []
    coords = []

    for i_idx, i in enumerate(range(0, H - p + 1, stride)):
        for j_idx, j in enumerate(range(0, W - p + 1, stride)):
            patch = image[i:i+p, j:j+p, :].reshape(-1)  # Flatten (p*p*C,)
            patch = (patch - mu) / stdi
            batch_patches.append(patch)
            coords.append((i_idx, j_idx))
            idx += 1

            if idx % batch_size == 0 or (i + stride > H - p and j + stride > W - p):
                batch_patches_np = np.array(batch_patches)
                labels, distances = model.predict(batch_patches_np, ret_dist=True)

                for (ii, jj), label, dist in zip(coords, labels, distances):
                    labels_out[ii, jj] = label
                    distances_out[ii, jj] = dist

                # Reset for next batch
                batch_patches = []
                coords = []

    return labels_out, distances_out
def predict_nonoverlapping(image, model, mu, stdi, p=32, batch_size=1024):
    H, W, C = image.shape
    out_H = H // p
    out_W = W // p

    labels_out = np.zeros((out_H, out_W), dtype=np.int32)
    distances_out = np.zeros((out_H, out_W, 2), dtype=np.float32)

    idx = 0
    batch_patches = []
    coords = []

    for i in range(0, H - p + 1, p):
        for j in range(0, W - p + 1, p):
            patch = image[i:i+p, j:j+p, :].reshape(-1)  # Flatten (p*p*C,)
            patch = (patch - mu) / stdi
            batch_patches.append(patch)
            coords.append((i // p, j // p))
            idx += 1

            if idx % batch_size == 0 or (i + p >= H and j + p >= W):
                batch_patches_np = np.array(batch_patches)
                labels, distances = model.predict(batch_patches_np, ret_dist=True)

                for (ii, jj), label, dist in zip(coords, labels, distances):
                    labels_out[ii, jj] = label
                    distances_out[ii, jj] = dist

                # Reset for next batch
                batch_patches = []
                coords = []

    return labels_out, distances_out
def upsample_distances(distances_out, target_shape):
    """
    Upsample a (H, W, C) distances array to match the target shape (H_full, W_full).
    Each channel is resized independently using bilinear interpolation.
    """
    H_full, W_full = target_shape
    C = distances_out.shape[2]
    upsampled = np.zeros((H_full, W_full, C), dtype=np.float32)

    for c in range(C):
        upsampled[..., c] = cv2.resize(distances_out[..., c], (W_full, H_full), interpolation=cv2.INTER_LINEAR)

    return upsampled

def slide_gialvq(orig,mu, stdi,  p, model, rgb2x=None):
    image = np.copy(orig)
    if rgb2x is not None:
        image = rgb2x(image)
    image = np.pad(image, ((p // 2, p // 2), (p // 2, p // 2), (0, 0)), mode='symmetric')
    # patches = np.lib.stride_tricks.sliding_window_view(image, window_shape=(p, p), axis=(0,1))
    labels, distances=predict_half_overlapping(image, model, mu, stdi, p=p)

    # patches = (patches.reshape(-1, p * p * 3) - mu)/stdi
    # labels, distances = model.predict(patches, ret_dist=True)
    probabilities = distances.reshape((distances.shape[0], -1, 2, model.prototypes_per_class)).min(-1)
    probabilities = softmin(probabilities, 0.001)
    probabilities = probabilities[:,:, 1]  # for now consider probability only for mold
    # distances = np.min(distances, axis=2)

    H = orig.shape[0]
    W = orig.shape[1]
    labels = cv2.resize(labels, (H,W), interpolation=cv2.INTER_NEAREST)
    probabilities = cv2.resize(probabilities, (H,W), interpolation=cv2.INTER_NEAREST)
    distances = upsample_distances(distances, (H, W))  # original image shape

    if p%2 == 0:
        new_H = H + 1
        new_W = W + 1
    else:
        new_H = H
        new_W = W

    labels = labels.reshape(new_H, new_W)#new_H//p, new_W//p
    distances = distances.reshape(new_H, new_W,2)
    probabilities = probabilities.reshape(new_H, new_W)

    return labels, distances, probabilities


def segment_gialvq(img,mu, stdi, gialvq):
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, 2),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.15
                     )

    grid[0].imshow(img[:,:,0:3])
    cmap = matplotlib.colors.ListedColormap(["peachpuff", "salmon", "red"])
    imgc = np.zeros((img[:,:,0:3].shape[0],img[:,:,0:3].shape[1],3))
    # imgc = imgc[:img.shape[0], :img.shape[1], :] 
    labels_cialvq_charting, d, prob1 = slide_gialvq(img, mu, stdi,55, gialvq)
    mold_pixels_bool_idxs = labels_cialvq_charting == 0
    idxs_eggs = labels_cialvq_charting == 1
    # imgc = np.zeros((87, 104, 3))
    imgc[idxs_eggs, 2] = 255
    imgc[idxs_eggs, 0] = 0
    imgc[idxs_eggs, 1] = 0
    imgc[mold_pixels_bool_idxs, :] = 255 * cmap(prob1[mold_pixels_bool_idxs])[:, 0:3]
    mappbl = grid[1].imshow(imgc, cmap=cmap)

    cbar = grid[1].cax.colorbar(mappbl, cmap=cmap)
    cbar.ax.set_yticks([0, 255])
    cbar.ax.set_yticklabels(['0', '1'])

    grid[0].set_xticks([])
    grid[1].set_xticks([])
    grid[0].set_yticks([])
    grid[1].set_yticks([])
    plt.axis('off')

    grid[0].set_title("Input", fontsize='large')
    grid[1].set_title("GMLVQ (sliding window)", fontsize='large')
    plt.tight_layout()
    plt.show()


def build_informed_tree(gialvq, x_train, img, patch_sz, preset_alphas=False):
    d, iqrs = gialvq.dist_to_protos(x_train, [75, 25])
    a = AlphaTree(img, patch_sz)
    if preset_alphas:
        alphas = iqrs.flatten()
    else:
        alphas = []
    a.build(alphas, gialvq, labels=None, alpha_start=0)
    return a, iqrs


# roses data set specific function (work in progress)
def cut_tree(atree, iqrs, cls, show_mode='default'):
    if type(cls) is int:
        if cls == 0:
            t = iqrs[0, 0, 0]
            s = "(within mold)"
        elif cls == 1:
            t = iqrs[1, 1, 0]
            s = "(within eggs)"
        else:
            t = iqrs[2, 2, 0]
            s = "(within healthy)"
    else:
        i, j, k = cls
        t = iqrs[i, j, k]
        s = str(t)

    if show_mode == 'default':
        fig, ax = plt.subplots()
        cmap = matplotlib.colormaps['twilight']
        res = atree.filter(t)
        u = np.unique(res).astype(int)

        im = ax.imshow(res, interpolation='none', cmap=ListedColormap(cmap(np.linspace(0.15, 0.95, len(u)))))
        plt.title(r"Cut at $Q_3$ %s" % s)
        plt.yticks([])
        plt.xticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad='2%')
        cb = plt.colorbar(im, cmap=cmap, cax=cax)
        cb.set_ticks([0.5, u[-1] - 0.5])
        cb.ax.set_yticklabels([0, len(u) - 1])
        plt.tight_layout()
        plt.show()
        return res
    elif show_mode == 'alpha':
        fig, ax = plt.subplots()
        cmap = matplotlib.colormaps['Greens']
        res = atree.filter2(t)

        im = ax.imshow(np.ln(res), interpolation='none', cmap=cmap)
        plt.title(r"Cut at $Q_3$ %s" % s)
        plt.yticks([])
        plt.xticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad='2%')
        cb = plt.colorbar(im, cmap=cmap, cax=cax)

        plt.tight_layout()
        plt.show()
        return res

    else:
        fig, ax = plt.subplots()
        res, cls_count = atree.filter3(t)

        res = res.astype(int)
        cls_count = cls_count.astype(int)
        u = np.unique(res)

        r = matplotlib.colormaps['Reds']
        b = matplotlib.colormaps['Blues']
        g = matplotlib.colormaps['Greens']
        classes = [r, b, g]
        class_colors = [c(np.linspace(0.3, 1, n)) for c, n in zip(classes, cls_count) if n > 0]
        colors = np.vstack(class_colors)
        cmap = matplotlib.colors.ListedColormap(colors)
        im = ax.imshow(res, cmap=cmap, interpolation='none')
        plt.title(r"Cut at $Q_3$ %s" % s)
        plt.yticks([])
        plt.xticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad='2%')

        cb = plt.colorbar(im, cmap=cmap, cax=cax)

        cb.set_ticks([0.5, u[-1] - 0.5])
        cb.ax.set_yticklabels([0, len(u) - 1])
        plt.tight_layout()
        plt.show()
        return res, cls_count
