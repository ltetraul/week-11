import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

#load diamonds dataset
diamonds = sns.load_dataset('diamonds')

#keep numerical columns only
diamonds_num = diamonds.select_dtypes(include=[np.number])

#define kmeans function
def kmeans(X, k):
    """
    Performs kmeans clustering on numerical array X.
    Returns (centroids, labels)
    """
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    centroids = model.cluster_centers_
    labels = model.labels_
    return centroids, labels


#define kmeans_diamonds function
def kmeans_diamonds(n, k):
    """
    Run kmeans clustering on first n rows of numeric diamonds dataset
    """
    #select first n rows and convert to array
    X = diamonds_num.head(n).to_numpy()

    #run kmeans function
    centroids, labels = kmeans(X, k)
    return centroids, labels

from time import time

#define kmeans_timer function
def kmeans_timer(n, k, n_iter=5):
    """
    Runs kmeans_diamonds exactly n_iter times and returns
    the average runtime in seconds.
    """
    runtimes = []

#run kmeans_diamonds n_iter times
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        elapsed = time() - start
        runtimes.append(elapsed)

#compute average runtime
    avg_time = sum(runtimes) / len(runtimes)
    return avg_time