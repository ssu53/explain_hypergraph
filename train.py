import torch

from train_utils import train_eval_loop_many
from load_coraca import get_coraca_hypergraph
from models.hypergraph_models import HyperGCN, HyperResidGCN



def put_hgraph_attributes_on_device(hgraph, device) -> None:
    hgraph.train_mask = hgraph.train_mask.to(device)
    hgraph.val_mask = hgraph.val_mask.to(device)
    hgraph.test_mask = hgraph.test_mask.to(device)
    hgraph.x = hgraph.x.to(device)
    hgraph.y = hgraph.y.to(device)
    hgraph.H = hgraph.H.to(device)
    hgraph.edge_index = hgraph.edge_index.to(device)



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hgraph_coraca = get_coraca_hypergraph(split=[0.5, 0.25, 0.25], split_seed=3)
    put_hgraph_attributes_on_device(hgraph_coraca, device)


    num_layers = 4
    alpha = 0.2
    beta = None # sweep
    dropout = 0.5


    for beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:

        results = train_eval_loop_many(
            nruns=10,
            model_class=HyperResidGCN,
            model_args=dict(input_dim=1433, hidden_dim=128, output_dim=7, num_layers=num_layers, alpha=alpha, beta=beta, dropout=dropout),
            hgraph=hgraph_coraca,
            train_mask=hgraph_coraca.train_mask,
            val_mask=hgraph_coraca.val_mask,
            test_mask=hgraph_coraca.test_mask,
            lr=0.01,
            num_epochs=80,
            verbose=False,
            save_dir='training_results/hgcn_resid',
            save_name=f'coraca_{num_layers}layer_lr1e-2_alpha{alpha}_beta{beta}_dropout{dropout}',
            device=device,
        )



