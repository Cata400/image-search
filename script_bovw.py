from utils import *

dataset_path = 'CorelDB'
seed = 42
class_names = os.listdir(dataset_path)

img_train, img_test, labels_train, labels_test = get_data(dataset_path)

np.random.seed(seed)
img_train, labels_train = np.random.permutation(img_train), np.random.permutation(labels_train)
img_test, labels_test = np.random.permutation(img_test), np.random.permutation(labels_test)

print(img_train.shape, img_test.shape, labels_train.shape, labels_test.shape)
print("Data reading done!")


# For a single image
orb = cv2.ORB_create(nfeatures=10000000, scoreType=cv2.ORB_FAST_SCORE)
sift = cv2.xfeatures2d.SIFT_create()
kp = orb.detect(img_train[0], None)
kp, des = orb.compute(img_train[6], kp)
img = cv2.drawKeypoints(img_train[6], kp, None, color=(0, 255, 0), flags=0)

plt.figure(), plt.imshow(img), plt.title('ORB keypoints'), plt.axis('off'), plt.show()

# For all images
### Extract keypoints and descriptors for test images
train_kp, train_des = [[], []], [[], []]
for i, img in enumerate(img_train):
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    
    if kp is not None:
        train_kp[0].extend(kp)
        train_kp[1].extend([i] * len(kp))
    if des is not None:
        train_des[0].extend(des)
        train_des[1].extend([i] * len(des))
    
train_data = np.array(train_des[0], dtype=object)
print(train_data.shape)

test_kp, test_des = [[], []], [[], []]
for i, img in enumerate(img_test):
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)

    if kp is not None:
        test_kp[0].extend(kp)
        test_kp[1].extend([i] * len(kp))
    if des is not None:
        test_des[0].extend(des)
        test_des[1].extend([i] * len(des))
    
test_data = np.array(test_des[0], dtype=object)
print(test_data.shape)
print("Feature extraction done!")

### Clustering 
clusters = 400
kmeans = KMeans(n_clusters=clusters, random_state=seed)
kmeans.fit(train_data)
kmeans_name = 'kmeans_des_clustering_sift_10clase.sav'
pickle.dump(kmeans, open(kmeans_name, 'wb'))
print("Clustering done!")

### Descriptors to histograms
train_hist = np.zeros((len(img_train), clusters), dtype=np.float32)
for i, img in enumerate(img_train):
    img_train_des = train_data[find_indeces(train_des[1], i)]
    
    if img_train_des.shape[0] != 0:
        clusters_pred = kmeans.predict(img_train_des)
        
        for cp in clusters_pred:
            train_hist[i, cp] += 1
        
### Classifying clusters
knn = NearestNeighbors(n_neighbors=5)
knn.fit(train_hist)
knn_name = 'knn_hist_clustering_sift_10clase.sav'
pickle.dump(knn, open(knn_name, 'wb'))
print("Classifying done!")
        
### Testing
test_hist = np.zeros((len(img_test), clusters), dtype=np.float32)
for i, img in enumerate(img_test):
    img_test_des = test_data[find_indeces(test_des[1], i)]
    
    if img_test_des.shape[0] != 0:
        clusters_pred = kmeans.predict(img_test_des)
        for cp in clusters_pred:
            test_hist[i, cp] += 1
            
        ### Predicting
        distances, indices = knn.kneighbors(test_hist[i].reshape(1, -1))
        distance_dict = {}
        for k, idx in enumerate(indices[0]):
            distance_dict[idx] = distances[0][k]
            
        print(f"Test image: {i} \t Closest images: distances {distance_dict}")
        fig, ax = plt.subplots(1, 6, figsize=(4, 4))
        ax[0].imshow(img), ax[0].set_title('Test'), ax[0].axis('off')
        for j in range(1, 6):
            ax[j].imshow(img_train[indices[0][j - 1]]), ax[j].set_title('Closest {}'.format(j)), ax[j].axis('off')
        plt.show()
        
        
        
