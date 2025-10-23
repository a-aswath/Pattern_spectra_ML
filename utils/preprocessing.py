import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import skimage.io
from numpy import copy
import cv2
from skimage.transform import resize

from sklearn.feature_extraction import image as sk_img
from utils.io_management import save_preprocessed_data


def preprocess(ds_train_path, ds_test_path=None, parent_dir= None, resize_to=False, resize_shape=(128, 128),
               extract_patches=True, patch_size=15,
               n_patches=200, rnd_state=11, as_gray=False, ds_name="data"):

    """
    Note, the following file structure is required:
        data/raw/DS_NAME
            /train
                /CLASS_1_NAME
                ...
                /CLASS_N_NAME
            /test
                /CLASS_1_NAME
                ...
                /CLASS_N_NAME
    Off-line preprocessor for raw RGB images
    :param ds_test_path: path to the test data (each class should be in separate folder)
    :param ds_train_path: path to the training data (each class should be in separate folder)
    :param n_patches: the number of patches per image (if applicable)
    :param patch_size: the size of a patch (if applicable)
    :param extract_patches: if True random patches are extracted from the image
    :return: saves the processed data and labels in two separate .npy files at "./data/processed/ds"
    """
    # os.mkdir(os.path.join(parent_dir, "data\preprocessed", ds_name))
    class_names = []
    with open(os.path.join(parent_dir, "data\preprocessed", ds_name, 'info.txt'), 'w') as f:
        f.write('Name: %s\n' % ds_name)
        if extract_patches:
            f.write('Patches per image: %d\n' % n_patches)
            f.write('Patch size: %d\n' % patch_size)
            f.write('Random state: %d\n' % rnd_state)
    for test, ds_path in enumerate([ds_train_path, ds_test_path]):
        preprocessed_ = []
        labels_ = []
        label = 0
        for class_dir in os.listdir(ds_path):
            if not test:
                class_names.append(class_dir)
            for img_id in os.listdir(os.path.join(ds_path, class_dir)):
                img = skimage.io.imread(os.path.join(ds_path, class_dir, img_id), as_gray=as_gray)
                if img.shape[-1] == 4:
                        img = img[:, :, :3]  # Drop the alpha channel
                if not as_gray:
                    # skimage loads in RGB → convert to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # Convert BGR to RGB (redundant, but follows your original step)
                    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # Convert BGR to HSV and cast to float for hue normalization
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                    image_lab = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2Lab)

                    # Normalize hue to range [0, 1] based on 359° range (if that's your goal)
                    hsv[..., 0] *= 1.0 / 179.0
                    img= rgb#np.concatenate((rgb, hsv), 2)
                if resize_to:
                    img = resize(img, resize_shape)
                if extract_patches:
                    patches = sk_img.extract_patches_2d(img, (patch_size, patch_size), max_patches=n_patches,
                                                        random_state=rnd_state)
                    for patch in patches:
                        vectorized = vectorize(patch, as_gray)
                        preprocessed_.append(vectorized)
                        labels_.append(label)
                else:
                    
                    vectorized = vectorize(img, as_gray)
                    preprocessed_.append(vectorized)
                    labels_.append(label)
            label = label + 1
        preprocessed = np.array(preprocessed_)
        labels = np.array(labels_)
        save_preprocessed_data(preprocessed, labels, 'croptimal_mondial_55', test, parent_dir)
        with open(os.path.join(parent_dir, "data\preprocessed", ds_name, 'info.txt'), 'a') as f:
            if not test:
                f.write('Number of classes: %d\n' % len(np.unique(labels)))
                for c in range(len(class_names)):
                    f.write('{%d : %s}\n' % (c, class_names[c]))
                f.write('Train size: %d\n' % len(labels))
            else:
                f.write('Test size: %d\n' % len(labels))
        if ds_test_path is None:
            return


def vectorize(img, as_gray=False):
    """
    Off-line preprocessing step
    :param img: pxpx3 image in Fourier/spatial domain
    :return: v
    """
    if not as_gray:
        img = img.transpose(2, 0, 1)
    return img.flatten()


def preproces_satimage(data, train=True):
    h = 4
    n = 3 * 3
    samples = data[:, 0:data.shape[1] - 1]
    labels = data[:, data.shape[1] - 1:]
    new_labels = copy(labels)
    d = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5}
    for k, v in d.items(): new_labels[labels == k] = v

    col_idx_ord = []
    for band in range(h):
        for i in range(n):
            col_idx_ord.append(band + i * h)
    samples = (samples[:, col_idx_ord])
    if train:
        np.save("data/satimage/samples_train.npy", samples)
        np.save("data/satimage/labels_train.npy", new_labels.ravel().astype(int))
    else:
        np.save("data/satimage/samples_test.npy", samples)
        np.save("data/satimage/labels_test.npy", new_labels.ravel().astype(int))


def preprocess_roses(ds_train_path, ds_test_path=None,
                     extract_patches=True, patch_size=15,
                     n_patches=200, rnd_state=11, as_gray=False, ds_name="data", rgb2x=None):
    """
    extracts only if img is not of the patch size
    """
    os.mkdir(os.path.join("data", ds_name))
    class_names = []
    with open(os.path.join(".", "data", ds_name, 'info.txt'), 'w') as f:
        f.write('Name: %s\n' % ds_name)
        if rgb2x is None:
            f.write('Colour space: rgb\n')
        else:
            f.write('Colour space: %s\n' % str(rgb2x))
        if extract_patches:
            f.write('Patch size: %d\n' % patch_size)
            f.write('Random state: %d\n' % rnd_state)
    for test, ds_path in enumerate([ds_train_path, ds_test_path]):
        preprocessed_ = []
        labels_ = []
        label = 0
        for class_dir in os.listdir(ds_path):
            if not test:
                class_names.append(class_dir)
            for img_id in os.listdir(os.path.join(ds_path, class_dir)):
                img = skimage.io.imread(os.path.join(ds_path, class_dir, img_id), as_gray=as_gray)

                if rgb2x is not None:
                    img = rgb2x(img)
                if img.shape[0] != patch_size and img.shape[1] != patch_size:
                    patches = sk_img.extract_patches_2d(img, (patch_size, patch_size), max_patches=n_patches,
                                                        random_state=rnd_state)
                    for patch in patches:
                        vectorized = vectorize(patch, as_gray)
                        preprocessed_.append(vectorized)
                        labels_.append(label)
                else:
                    vectorized = vectorize(img, as_gray)
                    preprocessed_.append(vectorized)
                    labels_.append(label)
            label = label + 1
        preprocessed = np.array(preprocessed_)
        labels = np.array(labels_)
        cl_distr = np.bincount(labels)
        save_preprocessed_data(preprocessed, labels, ds_name, test)
        with open(os.path.join(".", "data", ds_name, 'info.txt'), 'a') as f:
            if not test:
                f.write('Number of classes: %d\n' % len(np.unique(labels)))
                for c in range(len(class_names)):
                    f.write('{%d : %s (%d train patches)}\n' % (c, class_names[c], cl_distr[c]))
            else:
                for c in range(len(class_names)):
                    f.write('{%d : %s (%d test patches)}\n' % (c, class_names[c], cl_distr[c]))
        if ds_test_path is None:
            return
        
# import os
# import sys

# # Get the absolute path of the parent directory (where data is located)
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(parent_dir)  # Add parent directory to Python path

# # from utils.io_management import save_preprocessed_data  # Now this import should work

# # Use absolute paths for data directories
# ds_name="croptimal_mondial"

# ds_train_path = os.path.join(parent_dir, "data", "raw", ds_name, "train")  # Required
# ds_test_path = os.path.join(parent_dir, "data", "raw", ds_name, "test")  # Optional

# # Ensure that paths in preprocess() use the correct directory
# preprocess(
#     ds_train_path=ds_train_path,
#     ds_test_path=ds_test_path,
#     parent_dir=parent_dir,
#     resize_to=True,
#     resize_shape=(55, 55),
#     extract_patches=False,
#     patch_size=55,
#     n_patches=20,
#     rnd_state=11,
#     as_gray=False,
#     ds_name=r"croptimal_mondial_55"
# )