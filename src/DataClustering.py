import matplotlib as plt
import numpy as np
from sklearn.cluster import  KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import sys
sys.path.append('./src')
import logger
import file_operatons

class Clutering:
    """
    Used to perform clustering task on the given dataset.
    This is auxillary process, not used in Final Model
    """

    def __init__(self):
        self.logger = logger.create_logger('Clustering_Pipeline')

    def best_cluster(self):
        score_=[]
        range_n_clusters = list(range(2,15))
        for n_clusters in range_n_clusters:
            cluster_iter = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = cluster_iter.fit_predict(self.data)
            silhouette_avg = silhouette_score(self.data, cluster_labels)
            score_.append((n_clusters, silhouette_avg))
        score_.sort(key=lambda x: x[1], reverse=True)

        best_k = score_[0][0]
        return best_k



    def make_cluster(self, data):

        self.data = data
        best_k = self.best_cluster()
        try:
            self.kmeans = KMeans(n_clusters = best_k, init='k-means++', random_state=42)

            self.y_cluster = self.kmeans.fit_predict(self.data)
            self.fileops=file_operatons.FileOperation()
            self.fileops.save_model(self.kmeans, 'KMeans')

            return self.y_cluster


        except Exception as e:
            self.logger.info('Error While Validating Filenames!')
            raise Exception()




