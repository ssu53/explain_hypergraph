import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from typing import Dict
import torch



def plot_activation_by_class(activ: torch.Tensor, class_label: torch.Tensor, fig_title: str = " Activations by Class", figsize=(7,5)) -> None:

    activ = activ.detach().cpu()
    class_label = class_label.detach().cpu()

    pca_model = PCA(n_components=2)
    activ_pca = pca_model.fit_transform(activ)

    x_pca, y_pca = np.split(activ_pca, 2, axis=1)
    x_pca = x_pca.squeeze(1)
    y_pca = y_pca.squeeze(1)
    assert x_pca.shape == y_pca.shape == (len(activ),)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(fig_title)
    pca_scat = ax.scatter(x_pca, y_pca, c=class_label, alpha=0.5, cmap='viridis')
    ax.legend(handles=pca_scat.legend_elements()[0], labels=list(np.unique(class_label)))
    plt.show()
    


def plot_activations_by_class(activations: Dict[str, torch.Tensor], class_label: torch.Tensor, figsize=(7,5)) -> None:
    """
    Plot 2D PCA.
    """

    class_label = class_label.detach().cpu()

    for layer_name in activations.keys():
        plot_activation_by_class(activations[layer_name], class_label, f"Activations by Class, Layer {layer_name}", figsize)



def plot_clusters(data: np.ndarray, labels: np.ndarray, num_clusters: int, fig_title: str, figsize=(7, 5)) -> None:

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(fig_title)

    for i in range(num_clusters):
        ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster {i}', alpha=0.5)

    ncol = 1
    if num_clusters > 20:
        ncol = int(num_clusters / 20) + 1
    ax.legend(ncol=ncol)
    plt.show()



def plot_activation_by_cluster(activ: torch.Tensor, num_clusters: int = 5, plot: bool = True, fig_title: str = "Activations by Cluster"):

    activ = activ.detach().cpu()

    pca_model = PCA(n_components=2) # visualise in 2D
    activ_pca = pca_model.fit_transform(activ)

    x_pca, y_pca = np.split(activ_pca, 2, axis=1)
    x_pca = x_pca.squeeze(1)
    y_pca = y_pca.squeeze(1)
    
    kmeans_model = KMeans(num_clusters, random_state=0, n_init='auto').fit(activ)
    cluster_labels = kmeans_model.predict(activ)

    if plot:
        plot_clusters(activ_pca, cluster_labels, num_clusters, fig_title)

    return kmeans_model



def binarise(x, thresh=0.5):
    return torch.where(x > thresh, 1.0, 0.0)



def plot_concepts(hgraph, model, num_clusters=6, binarise_concepts=False):

    # activations = {}
    # hook_handles = {}

    # for i in range(len(model.gcn_layers)):
    #     h = model.gcn_layers[i].register_forward_hook(get_activations(f"conv{i}", activations))
    #     hook_handles[f"conv{i}"] = h

    # with torch.no_grad():
    #     out = model(hgraph)

    # # remove hooks to freeze activations
    # for h in hook_handles.values(): h.remove()

    # activ, kmeans_model = plot_cluster_activations(activations, 'conv2', num_clusters=num_clusters)


    with torch.no_grad():
        out = model(hgraph)
        
    concepts = model.concepts.detach().cpu()
    if binarise_concepts: concepts = binarise(concepts)

    kmeans_model = plot_activation_by_class(concepts, hgraph.y)
    kmeans_model = plot_activation_by_cluster(concepts, num_clusters)

    return concepts, kmeans_model