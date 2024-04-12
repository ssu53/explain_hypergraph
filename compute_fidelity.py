# %%

import numpy as np
import torch
from train import get_single_run
import json
from easydict import EasyDict
from pathlib import Path

path = Path('train_results/zoo/allsettransformer/run1')
load_best = False

_, _, hgraph, model = get_single_run(path, device=torch.device("cpu"), load_best=load_best)

with torch.no_grad():
    logits = model(hgraph)
    y_pred = logits.argmax(dim=-1)
    class_label = hgraph.y

# %%

with open(path / 'explanation.json') as f:
    explanation = EasyDict(json.load(f))

expl_loss_relaxed = []
expl_loss_binarised = []
expl_class_relaxed = []
expl_class_binarised = []
compl_loss_relaxed = []
compl_loss_binarised = []
compl_class_relaxed = []
compl_class_binarised = []


for i,node_idx in enumerate(explanation.summary):
    assert i == int(node_idx)

    # explanation
    expl_loss_relaxed.append(explanation.summary[node_idx].explanation['loss/relaxed']) 
    expl_loss_binarised.append(explanation.summary[node_idx].explanation['loss/binarised']) 
    expl_class_relaxed.append(np.argmax(explanation.summary[node_idx].explanation['classprob/relaxed']))
    expl_class_binarised.append(np.argmax(explanation.summary[node_idx].explanation['classprob/binarised']))
    
    # complement
    compl_loss_relaxed.append(explanation.summary[node_idx].complement['loss/relaxed']) 
    compl_loss_binarised.append(explanation.summary[node_idx].complement['loss/binarised']) 
    compl_class_relaxed.append(np.argmax(explanation.summary[node_idx].complement['classprob/relaxed']))
    compl_class_binarised.append(np.argmax(explanation.summary[node_idx].complement['classprob/binarised']))


expl_loss_relaxed = torch.tensor(expl_loss_relaxed)
expl_loss_binarised = torch.tensor(expl_loss_binarised)
expl_class_relaxed = torch.tensor(expl_class_relaxed)
expl_class_binarised = torch.tensor(expl_class_binarised)
compl_loss_relaxed = torch.tensor(compl_loss_relaxed)
compl_loss_binarised = torch.tensor(compl_loss_binarised)
compl_class_relaxed = torch.tensor(compl_class_relaxed)
compl_class_binarised = torch.tensor(compl_class_binarised)

# %%

N = len(class_label)

fidplus_soft = torch.abs((y_pred == class_label).to(int) - (compl_class_relaxed == class_label).to(int)).sum() / N
fidminus_soft = torch.abs((y_pred == class_label).to(int) - (expl_class_relaxed == class_label).to(int)).sum() / N
fidplus_hard = torch.abs((y_pred == class_label).to(int) - (compl_class_binarised == class_label).to(int)).sum() / N
fidminus_hard = torch.abs((y_pred == class_label).to(int) - (expl_class_binarised == class_label).to(int)).sum() / N

print("relaxed (soft masks)")
print(f"fidelity+ {fidplus_soft:.3f} | fidelity- {fidminus_soft:.3f}")

print("binarised (hard masks)")
print(f"fidelity+ {fidplus_hard:.3f} | fidelity- {fidminus_hard:.3f}")
# %%
