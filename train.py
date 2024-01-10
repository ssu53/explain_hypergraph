import pickle
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from models import MyHyperGCN, HyperResidGCN
from train_utils import train_eval_loop
from hgraph.utils import put_hgraph_attributes_on_device, hgraph_to_dict
from hgraph.generate import make_hgraph



def get_model_class(model):

    if model == "MyHyperGCN":
        return MyHyperGCN
    elif model == "HyperResidGCN":
        return HyperResidGCN
    else:
        raise NotImplementedError



def save_stuff(cfg, train_stats, hgraph, model):

    path = Path(cfg.save_dir)
    if cfg.save_datestamp:
        path = path / f"_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    path.mkdir(exist_ok=True, parents=True)

    # save cfg
    dict_cfg = OmegaConf.to_container(cfg)
    with open(path / "cfg.json", "w") as f:
        json.dump(dict_cfg, f, indent=4)

    # save train_stats
    train_stats = pd.DataFrame.from_dict(train_stats)
    train_stats = train_stats.set_index("epoch")
    train_stats.to_csv(path / "train_stats.csv")

    # save hgraph
    dict_hgraph = hgraph_to_dict(hgraph)
    with open(path / "hgraph.pickle", "wb") as f:
        pickle.dump(dict_hgraph, f)
    
    # save model
    torch.save(model.state_dict(), path / "model")



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    hgraph = make_hgraph(cfg.hgraph)
    put_hgraph_attributes_on_device(hgraph, device)


    model_args = dict(cfg.model.model_params)
    model_args["input_dim"] = hgraph.x.shape[1]
    model_args["output_dim"] = cfg.hgraph.num_house_types * 3 + 1
    model = get_model_class(cfg.model.model)(**model_args)
    model.to(device)
    print(model)


    train_stats = train_eval_loop(
        model=model,
        hgraph=hgraph,
        train_mask=hgraph.train_mask,
        val_mask=hgraph.val_mask,
        test_mask=hgraph.test_mask,
        **cfg.train,
    )

    # train_stats = train_eval_loop_many(
    #     nruns=10,
    #     model_class=get_model_class(cfg.model.model),
    #     model_args=model_args,
    #     hgraph=hgraph,
    #     train_mask=hgraph.train_mask,
    #     val_mask=hgraph.val_mask,
    #     test_mask=hgraph.test_mask,
    #     lr=cfg.train.lr,
    #     num_epochs=cfg.train.num_epochs,
    #     verbose=cfg.train.verbose,
    #     device=device,
    # )

    save_stuff(cfg, train_stats, hgraph, model)




if __name__ == "__main__":
    main()
