import cv2
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def get_data(dataset_path):
    img_train, img_test, labels_train, labels_test = [], [], [], []
    
    for i, c in enumerate(os.listdir(dataset_path)):
        for j, file in enumerate(os.listdir(os.path.join(dataset_path, c))):
            if file.endswith('.jpg'):        
                img = cv2.imread(os.path.join(dataset_path, c, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if j % 10 == 0:
                    img_test.append(img)
                    labels_test.append(i)
                else:
                    img_train.append(img)
                    labels_train.append(i)
                                
    img_train, img_test, labels_train, labels_test = np.array(img_train), np.array(img_test), np.array(labels_train), np.array(labels_test)
    
    return img_train, img_test, labels_train, labels_test


def draw_keypoints(vis, keypoints, color = (0, 255, 0)):
    for kp in keypoints:
        x, y = kp.pt
        vis = cv2.circle(vis, (int(x), int(y)), 2, color)
                
    plt.imshow(vis), plt.title("ORB keypoints 2"), plt.show()


def find_indeces(list, value):
    return [i for i, x in enumerate(list) if x == value]

