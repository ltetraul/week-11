import numpy as np
from sklearn.cluster import KMeans

def kmeans(X, k):
    """
    Performs k-means clustering on numerical numpy array X.
    """
    #initialize kmeans model
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(X)

    #extract centroids and labels
    centroids = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_

    return centroids, labels