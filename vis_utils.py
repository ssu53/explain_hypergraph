import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from typing import Dict
import torch



def plot_activations_by_class(activations: Dict[str, torch.Tensor], class_label: torch.Tensor, figsize=(7,5)) -> None:
    """
    Plot 2D PCA.
    """

    class_label = class_label.detach().cpu()

    for layer_name in activations.keys():

        activ = activations[layer_name].detach().cpu()

        pca_model = PCA(n_components=2)
        activ_pca = pca_model.fit_transform(activ)

        x_pca, y_pca = np.split(activ_pca, 2, axis=1)
        x_pca = x_pca.squeeze(1)
        y_pca = y_pca.squeeze(1)
        assert x_pca.shape == y_pca.shape == (len(activ),)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{layer_name} activations")
        pca_scat = ax.scatter(x_pca, y_pca, c=class_label, alpha=0.5, cmap='viridis')
        ax.legend(handles=pca_scat.legend_elements()[0], labels=list(np.unique(class_label)))
        plt.show()



def plot_clusters(data: np.ndarray, labels: np.ndarray, num_clusters: int, layer_name: str, figsize=(7, 5)) -> None:

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(f'Clustered Activations of Layer {layer_name}')

    for i in range(num_clusters):
        ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster {i}', alpha=0.5)

    ncol = 1
    if num_clusters > 20:
        ncol = int(num_clusters / 20) + 1
    ax.legend(ncol=ncol)
    plt.show()



def plot_cluster_activations(activations: Dict[str, torch.Tensor], layer_name: str, num_clusters: int = 5):

    activ = activations[layer_name].detach().cpu()

    pca_model = PCA(n_components=2) # visualise in 2D
    activ_pca = pca_model.fit_transform(activ)

    x_pca, y_pca = np.split(activ_pca, 2, axis=1)
    x_pca = x_pca.squeeze(1)
    y_pca = y_pca.squeeze(1)
    
    kmeans_model = KMeans(num_clusters, random_state=0, n_init='auto').fit(activ)
    cluster_labels = kmeans_model.predict(activ)

    plot_clusters(activ_pca, cluster_labels, num_clusters, layer_name)

    return activ, kmeans_model