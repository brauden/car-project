import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')

if __name__ == '__main__':
    with open("../data/test_vectors.npy", "rb") as f:
        vectors = np.load(f)
    with open("../data/test_names.npy", "rb") as f:
        names = np.load(f)

    random_index = np.random.randint(0, vectors.shape[0])
    test_image = vectors[random_index]
    test_name = names[random_index]
    print(test_name)
    neighbors.fit(vectors)
    dist, indices = neighbors.kneighbors(test_image.reshape(1, -1))
    print(names[np.sort(indices[0])])

    with open("../data/similarity_model.pickle", "wb") as model:
        pickle.dump(neighbors, model)
