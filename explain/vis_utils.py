import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from typing import Dict, Union, List
import torch



def plot_activation_by_class(activ: torch.Tensor, class_label: Union[torch.Tensor, List], categorical_label: bool = True, fig_title: str = " Activations by Class", figsize=(7,5), verbose: bool = True) -> None:

    activ = activ.detach().cpu()
    if isinstance(class_label, torch.Tensor):
        class_label = class_label.detach().cpu()

    pca_model = PCA(n_components=2)
    activ_pca = pca_model.fit_transform(activ)

    x_pca, y_pca = np.split(activ_pca, 2, axis=1)
    x_pca = x_pca.squeeze(1)
    y_pca = y_pca.squeeze(1)
    assert x_pca.shape == y_pca.shape == (len(activ),)

    if verbose: 
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(fig_title)

        if categorical_label:

            class_label = np.array(class_label)
            _, sorted_idx = np.unique(class_label, return_index=True)
            class_to_ind = class_label[np.sort(sorted_idx)] # unique classes, in stable order
            class_to_ind = dict(zip(class_to_ind, range(len(class_to_ind))))
            colour = [class_to_ind[c] for c in class_label]
            
            pca_scat = ax.scatter(x_pca, y_pca, c=colour, alpha=0.5, cmap="Dark2_r")
            ax.legend(handles=pca_scat.legend_elements()[0], labels=class_to_ind.keys())

        else:
            pca_scat = ax.scatter(x_pca, y_pca, c=class_label, alpha=0.5, s=3, cmap='inferno_r')
            fig.colorbar(pca_scat, ax=ax)

        
        plt.show()
        # fig.savefig(f"{fig_title}.svg", format="svg", bbox_inches='tight')

    return pca_model
    


def plot_activations_by_class(activations: Dict[str, torch.Tensor], class_label: Union[torch.Tensor, List], categorical_label: bool = True, figsize=(7,5)) -> None:
    """
    Plot 2D PCA.
    """

    for layer_name in activations.keys():
        plot_activation_by_class(activations[layer_name], class_label, categorical_label, f"Activations by Class, Layer {layer_name}", figsize)



def plot_clusters(data: np.ndarray, labels: np.ndarray, num_clusters: int, fig_title: str, figsize=(7, 5)) -> None:

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(fig_title)

    for i in range(num_clusters):
        ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Concept {i}')

    ncol = 1
    if num_clusters > 20:
        ncol = int(num_clusters / 20) + 1
    ax.legend(ncol=ncol)
    plt.show()
    # fig.savefig(f"{fig_title}.svg", format="svg", bbox_inches='tight')



def plot_activation_by_cluster(activ: torch.Tensor, num_clusters: int = 5, fig_title: str = "Activations by Cluster", verbose: bool = True):

    pca_model = PCA(n_components=2) # visualise in 2D
    activ_pca = pca_model.fit_transform(activ)
    
    kmeans_model = KMeans(num_clusters, random_state=0, n_init='auto').fit(activ)
    cluster_labels = kmeans_model.predict(activ)

    if verbose:
        plot_clusters(activ_pca, cluster_labels, num_clusters, fig_title)

    return kmeans_model



def binarise(x, thresh=0.5, as_bool=True):
    if as_bool:
        return torch.where(x > thresh, True, False)
    return torch.where(x > thresh, 1.0, 0.0)



def plot_activation_by_binarise(activ, fig_title: str = "Activations by Binarisation", verbose: bool = True):

    labels = binarise(activ, as_bool=True)
    labels = [tuple(labels[i].tolist()) for i in range(labels.size(0))]
    codes = set(labels)
    mapping = {code:i for i,code in enumerate(codes)}
    labels = np.array([mapping[code] for code in labels])

    pca_model = PCA(n_components=2) # visualise in 2D
    activ_pca = pca_model.fit_transform(activ)

    if verbose:
        plot_clusters(activ_pca, labels, len(codes), fig_title)

    return labels



def plot_concepts(activ, labels, categorical_label: bool = True, num_clusters: int = 6, cluster_by_binarise: bool = False, fig_title: str = None, verbose: bool = True):

    pca_model = plot_activation_by_class(
        activ, labels, categorical_label, verbose=verbose, fig_title="Activations By Class" if fig_title is None else 
                                f"Activations By Class | {fig_title}")
    
    if cluster_by_binarise:
        labels = plot_activation_by_binarise(
            activ, fig_title="Activations By Binarisation" if fig_title is None else 
                            f"Activations By Binarisation | {fig_title}")
        kmeans_model = labels # assign this for return
    else:
        kmeans_model = plot_activation_by_cluster(
            activ, num_clusters, verbose=verbose, fig_title="Activations By Cluster" if fig_title is None else 
                                          f"Activations By Concept | {fig_title}")

    return kmeans_model, pca_model