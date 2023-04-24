import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat as load_matlab_element
from scipy.special import kl_div as kl_divergence
from scipy.signal import correlate2d
import time

from skimage.metrics import structural_similarity
from skimage.transform import resize, rotate
from skimage.feature import graycomatrix, graycoprops

from sklearn.model_selection import train_test_split

### FUNCTIONS
def loadmat(path):
    array = load_matlab_element(path)['file_list'].reshape(-1)
    list = [i[0] for i in array]
    return list


def load_img(filename, grayscale, bbox=None):
    img = plt.imread(filename)
    if img.max() < 2: img = np.uint8(255 * img)
    if grayscale: img = rgb2gray(img)
    if bbox:
        text = open('bbox/' + filename.replace('.jpg', '')).read()
        x_min = int(text.split('<xmin>')[1].split('</xmin>')[0])
        x_max = int(text.split('<xmax>')[1].split('</xmax>')[0])
        y_min = int(text.split('<ymin>')[1].split('</ymin>')[0])
        y_max = int(text.split('<ymax>')[1].split('</ymax>')[0])
        img = img[y_min:y_max, x_min:x_max]
    return img




get_label = lambda filename: "_".join(filename.split('-')[1:]).split('/')[0]

rgb2gray = lambda img: np.uint8(0.3 * img[:,:,0] + 0.6 * img[:,:,1] + 0.1 * img[:,:,2])

def histogram(x):
    if x.ndim == 2:
        h, _ = np.histogram(x, 256, range=(0, 255), density=True)
    elif x.ndim == 3:
        h_red, _ = np.histogram(x[:,:, 0], 256, range=(0, 255), density=True)
        h_green, _ = np.histogram(x[:, :, 1], 256, range=(0, 255), density=True)
        h_blue, _ = np.histogram(x[:, :, 2], 256, range=(0, 255), density=True)
        h = np.concatenate((h_red, h_green, h_blue))
    else:
        raise NotImplementedError("Dimensions are not compatible for a histogram")

    return h


def get_all_labels(element_list):
    all_labels = []
    for filename in element_list:
        label = get_label(filename)
        if label not in all_labels:
            all_labels.append(label)
    labels_dict = {label: idx for idx, label in enumerate(all_labels)}
    return labels_dict


def compute_kl_div(x, y):
    kl = kl_divergence(x, y)
    kl_value = kl[np.isfinite(kl)].sum()
    return kl_value


compute_mse = lambda x, y : np.mean((x - y)**2)

compute_mae = lambda x, y : np.mean(np.abs(x - y))

def compute_ssim(x, y):
    if x.ndim == 3:
        x_r = rgb2gray(x)
        y_r = rgb2gray(y)
    if x_r.shape == (120, 80):
        x_r = rotate(x_r, 90, resize=True, preserve_range=True)
    if y_r.shape == (120, 80):
        y_r = rotate(y_r, 90, resize=True, preserve_range=True)
    ssim_val = structural_similarity(x_r, y_r)
    return ssim_val


def compute_cross_corr(x, y):
    if x.ndim == 3:
        x1 = rgb2gray(x)
        y1 = rgb2gray(y)
    corr = correlate2d(x1, y1, boundary='symm')
    return corr.sum()



def get_encoding(img, method):
    if method == 'histogram':
        enc = histogram(img)
    elif method == 'binary_index':
        pass
    elif method == 'none':
        enc = img
    elif method == 'glcm':
        bins = list(range(10, 265, 10))
        img_digitized = np.digitize(img, bins)
        gcm = graycomatrix(img_digitized, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=len(bins)+1)
        # features = ['homogeneity', 'energy', 'correlation']
        features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        enc = []
        for feat in features:
            enc.append(graycoprops(gcm, feat))

    else:
        raise NotImplementedError('Requested method is not implemented !')
    return np.array(enc)


def get_train_data(train_list, grayscale=False, method='histogram', bbox=False):
    data = []
    for filename in train_list:
        img = load_img(filename, grayscale, bbox)
        d = get_encoding(img, method)
        data.append(d)
    return data


def load_and_split_data(path, seed=42, test_size=0.2, limit=80):
    file_list = []
    labels_list = []
    for i, folder in enumerate(os.listdir(path)):
        if i == limit:
            break
        for file in os.listdir(os.path.join(path,folder)):
            if file.endswith('.jpg'):
                file_list.append(os.path.join(path, folder, file))
                labels_list.append(i)

    np.random.seed(seed)
    x_train, x_test, y_train, y_test = train_test_split(file_list, labels_list, test_size=test_size, random_state=seed)

    return x_train, x_test, y_train, y_test




def get_similar_imgs_for_test(test_list, train_list, data, train_labels, test_labels,
                                criterion, grayscale=False, method='histogram', 
                                bbox=False, topk=5, show=True):

    global_ssims = []
    tp, fp, tn, fn = 0, 0, 0, 0
    count = 0


    for idx_test, filename in enumerate(test_list):
        img = load_img(filename, grayscale, bbox)
        img_copy = load_img(filename, False, bbox)
        enc1 = get_encoding(img, method)

        argmins = []
        min = 99999999999 if criterion in ['kl', 'mse', 'mae'] else -9999999999

        start = time.time()
        for i, enc2 in enumerate(data):
            
            if criterion == 'kl':
                loss = compute_kl_div(enc1, enc2)
            elif criterion == 'mse':
                loss = compute_mse(enc1, enc2)
            elif criterion == 'ssim':
                loss = compute_ssim(enc1, enc2)
            elif criterion == 'corr':
                loss = compute_cross_corr(enc1, enc2)
            else:  # defaults to MAE
                loss = compute_mae(enc1, enc2)

            if criterion in ['corr', 'ssim']:
                if loss > min:
                    argmins.append(i)
                    min = loss
            else:
                if loss < min:
                    argmins.append(i)
                    min = loss

        stop = time.time()
        imgs_sim = []
        if len(argmins) < topk:
            continue
        for i in range(topk):
            img_sim = load_img(train_list[argmins[-1 -i]], False, bbox)
            imgs_sim.append(img_sim)

        if show:
            _, axarr = plt.subplots(1, topk+1)
            axarr[0].imshow(img_copy)
            axarr[0].set_title('Searched image')
            axarr[0].set_xticks([])
            axarr[0].set_yticks([])
            for img_idx in range(topk):

                if img_idx == 0:
                    text = f'{img_idx+1}^st similar image'
                elif img_idx == 1:
                    text = f'{img_idx+1}^nd similar image'
                elif img_idx == 2:
                    text = f'{img_idx+1}^rd similar image'
                else:
                    text = f'{img_idx+1}^th similar image'

                axarr[img_idx+1].imshow(imgs_sim[img_idx])
                axarr[img_idx+1].set_title(text)
                axarr[img_idx+1].set_xticks([])
                axarr[img_idx+1].set_yticks([])
            plt.show()

        # ssim
        ssim_max = 0
        for i in range(topk):
            ssim = compute_ssim(img_copy, imgs_sim[i])
            if ssim > ssim_max and ssim >0.5:
                ssim_max = ssim
        # print(f"Max ssim: {ssim_max}")
        if ssim_max > 0:
            global_ssims.append(ssim_max)

        # acc, prec, rec
        ok_labels = 0
        for i in range(topk):
            if train_labels[argmins[-1 -i]] == test_labels[idx_test]:
                ok_labels += 1
        
        count += topk

        tp += ok_labels / topk
        fp += (topk - ok_labels) / topk



    print(f"Accuracy: {tp / count}")
    print(f"Precision: {tp / (tp + fp)}")
    print(f"Global avg ssim: {np.mean(global_ssims)}")

            

### MAIN
if __name__ == '__main__':

    DATA_PATH = 'Data/CorelDB'
    DATASET_LIMIT = 10
    LOSS = 'kl'
    GRAYSCALE = True
    METHOD = 'glcm'
    BBOX = False
    TOPK = 1
    SHOW = False

    train_list, test_list, train_labels, test_labels = load_and_split_data(DATA_PATH, limit=DATASET_LIMIT)
    print(f"Train size: {len(train_list), len(train_labels)}")
    print(f"Test size: {len(test_list), len(test_labels)}")

    start = time.time()
    data = get_train_data(train_list, GRAYSCALE, METHOD, BBOX)
    stop = time.time()
    print("Avg indexing time:", format(100 * (stop-start) / len(train_list), '.6f'), 'ms')

    get_similar_imgs_for_test(test_list, train_list, data, train_labels, test_labels,
                                LOSS, GRAYSCALE, METHOD, BBOX, TOPK, SHOW)


    
