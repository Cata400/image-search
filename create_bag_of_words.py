from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.model_selection import train_test_split
from sklearn.decomposition import MiniBatchDictionaryLearning

import os
import numpy as np
import matplotlib.pyplot as plt


def parse_folder(folder_path):
    return sorted([os.path.join(folder_path, file) for file in \
        os.listdir(folder_path) if file.endswith('.jpg')])


def parse_dataset(base_path, max_folders=-1):
    all_files = []
    all_labels = []
    for i, folder in enumerate(sorted(os.listdir(base_path))):
        folder_path = os.path.join(base_path, folder)
        curr_files = parse_folder(folder_path)
        all_files.extend(curr_files)
        all_labels.extend([i for _ in range(len(curr_files))])
        if i+1 == max_folders:
            break
    return all_files, all_labels


def extract_patches(img_path, patch_size=(5,5), overlap=0.5, gray=True, vector=True):
    img = plt.imread(img_path)
    if gray:
        img = 0.3 * img[..., 0] + 0.6 * img[..., 1] + 0.1 * img[..., 2]
    patches = extract_patches_2d(img, patch_size, max_patches=overlap)
    if vector:
        patches = patches.reshape((patches.shape[0], -1))
    return patches
    

if __name__ == '__main__':
    
    PATCH_SIZE = (5,5)
    MAX_VOCAB = 1000
    MAX_ITER = 20


    all_files, all_labels = parse_dataset('CorelDB', max_folders=-1)

    train_files, test_files, train_labels, test_labels = train_test_split(all_files, all_labels, test_size=0.1, random_state=42)


    mbdl = MiniBatchDictionaryLearning(n_components=MAX_VOCAB, max_iter=MAX_ITER)
    for file in train_files:
        patches = extract_patches(file, PATCH_SIZE)
        print(f"Started learning for file {file}...")
        mbdl.partial_fit(patches)
        print(f"Finished learning for file {file}")
        


