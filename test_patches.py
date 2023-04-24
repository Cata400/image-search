import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning    


if __name__ == '__main__':
    
    IMG_PATH = 'lena.jpeg'
    PATCH_SIZE = (3, 3)
    SEED = 42
    
    N_COMPONENTS = 10
    BATCH_SIZE = 300
    
    
    
    img = plt.imread(IMG_PATH)
    img = np.uint8(0.3 * img[...,0] + 0.6 * img[...,1] + 0.1 * img[...,2])
    plt.figure(), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title("Original image"), plt.show()
    img = img / 255 
    h, w = img.shape
    
    
    patches = extract_patches_2d(img, PATCH_SIZE, random_state=SEED)
    # for i, patch in enumerate(patches[25000:25020]):
        # plt.figure(), plt.imshow(patch), plt.title("Patch" + str(i)), plt.show()
    patches = np.reshape(patches, (patches.shape[0], -1))

    ### BoW Learning
    dict_learning = MiniBatchDictionaryLearning(N_COMPONENTS, 
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                # n_iter=200
    )
    
    BoW = dict_learning.fit(patches).components_
        
    plt.figure()
    for i, word in enumerate(BoW.T[:25]):
        plt.subplot(5, 5, i+1)
        plt.imshow(word[:-1].reshape(PATCH_SIZE), cmap='gray', interpolation="none")
        plt.xticks(())
        plt.yticks(())
        plt.colorbar()
    plt.suptitle("Example of extracted words")
    plt.show()
    

    patches_transformed = dict_learning.transform(patches)
    patches_transformed = patches_transformed[:, :-1]
    patches_transformed = patches_transformed.reshape(-1, *PATCH_SIZE)
        
    img_reconstruct = reconstruct_from_patches_2d(patches_transformed, (h, w))
    img_reconstruct = np.uint8(255 * img_reconstruct)
    plt.figure(), plt.imshow(img_reconstruct, cmap='gray', vmin=0, vmax=255),
    plt.title("Reconstructed image"), plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        