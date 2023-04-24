import numpy as np
import matplotlib.pyplot as plt
import os


def get_shape_of_img(img_path):
    img = plt.imread(img_path)
    return img.shape


if __name__ == '__main__':

    DATASET_PATH = 'CPPSMS\CorelDB'
    SEED = 42

    all_files = []
    class_distribution = {}
    for folder in sorted(os.listdir(DATASET_PATH)):
        class_distribution[folder] = len(os.path.join(DATASET_PATH, folder))
        for file in [i for i in os.listdir(os.path.join(DATASET_PATH, folder)) if i.endswith('.jpg')]:
            file_path = os.path.join(DATASET_PATH, folder, file)
            all_files.append(file_path)

    all_shapes = [get_shape_of_img(i) for i in all_files]

    all_shapes_hist = {(80, 120, 3): 0, (120, 80, 3): 0}
    for shape in all_shapes:
        all_shapes_hist[shape] += 1

    all_shapes_hist = {key: val/len(all_shapes) for key, val in all_shapes_hist.items()}
    class_distribution = {key: val/10800 for key, val in class_distribution.items()}

    ### Save fig of shapes
    x = [str(i[0]) + ' x ' + str(i[1]) + ' x ' + str(i[2]) for i in all_shapes_hist.keys()]
    y = [i*100 for i in all_shapes_hist.values()]
    title = 'Image shape distribution'
    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel('Image shape (H x W x C)')
    plt.ylabel('Percetage [%]')
    plt.savefig(os.path.join(os.getcwd(), 'CPPSMS', 'poze', title))


    ### Save fig of class distribution
    x = ["_".join(i.split('_')[1:]) for i in class_distribution.keys()]
    y = [i*100 for i in class_distribution.values()]
    show_limit = 15
    title = 'Class distribution'
    plt.figure()
    plt.barh(x[show_limit:show_limit+show_limit], y[show_limit:show_limit+show_limit])
    plt.title(title)
    plt.ylabel('Class name')
    plt.xlabel('Percetage [%]')
    plt.savefig(os.path.join(os.getcwd(), 'CPPSMS', 'poze', title))


    ### Save fig of random samples
    no_of_pics = (5, 5)
    imgs_idx = np.random.choice(all_files, size=np.prod(no_of_pics))
    title = 'Examples of images'
    _, ax = plt.subplots(*no_of_pics)
    for i in range(no_of_pics[0]):
        for j in range(no_of_pics[1]):
            path = imgs_idx[i * no_of_pics[1] + j]
            img = plt.imread(path)
            ax[i][j].imshow(img)
            ax[i][j].set_axis_off()
            mini_title = "_".join(path.split(os.sep)[-2].split('_'))
            ax[i][j].set_title(mini_title)   
    plt.suptitle(title)
    plt.savefig(os.path.join(os.getcwd(), 'CPPSMS', 'poze', title))




