# %%

import pickle
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hgraph.generate import make_hgraph
from hgraph.utils import hgraph_to_dict



@hydra.main(version_base=None, config_path="configs", config_name="generate_hgraph")
def main(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    n = cfg.num_graphs
    path = Path(cfg.save_dir)
    path.mkdir(exist_ok=False, parents=True)


    # save cfg
    dict_cfg = OmegaConf.to_container(cfg)
    with open(path / "cfg.json", "w") as f:
        json.dump(dict_cfg, f, indent=4)


    print(f"Generating {n} instances of this hypergraph and saving to {path}...")


    for i in range(n):

        hgraph = make_hgraph(cfg.hgraph)

        # save hgraph
        dict_hgraph = hgraph_to_dict(hgraph)
        with open(path / f"hgraph{i}.pickle", "wb") as f:
            pickle.dump(dict_hgraph, f)



if __name__ == "__main__":
    main()
