# %%

import numpy as np
import torch
from train import get_single_run
import json
from easydict import EasyDict
from pathlib import Path

# path = Path('train_results/zoo/allsettransformer/run1')
# load_best = False
# path = Path('train_results/coauthor_cora/allsettransformer/run1')
# load_best = True
path = Path('train_results/randhouse_v3/allsettransformer/hgraph0/run0')
load_best = True

_, _, hgraph, model = get_single_run(path, device=torch.device("cpu"), load_best=load_best)

with torch.no_grad():
    logits = model(hgraph)
    probs = logits.softmax(dim=-1)
    y_pred = logits.argmax(dim=-1)
    class_label = hgraph.y

# %%

with open(path / 'explanation.json') as f:
    explanation = EasyDict(json.load(f))

assert explanation.config.load_fn == str(path)
assert explanation.config.load_best == load_best

expl_loss_human = []
expl_loss_raw = []
expl_loss_post = []

expl_probs_human = []
expl_probs_raw = []
expl_probs_post = []

expl_class_human = []
expl_class_raw = []
expl_class_post = []

compl_loss_human = []
compl_loss_raw = []
compl_loss_post = []

compl_class_human = []
compl_class_raw = []
compl_class_post = []

compl_probs_human = []
compl_probs_raw = []
compl_probs_post = []


for i,node_idx in enumerate(explanation.summary):
    assert i == int(node_idx)

    # explanation
    expl_loss_human.append(explanation.summary[node_idx].explanation['loss/human'])
    expl_loss_raw.append(explanation.summary[node_idx].explanation['loss/raw']) 
    expl_loss_post.append(explanation.summary[node_idx].explanation['loss/post']) 
    
    expl_class_human.append(np.argmax(explanation.summary[node_idx].explanation['classprob/human']))
    expl_class_raw.append(np.argmax(explanation.summary[node_idx].explanation['classprob/raw']))
    expl_class_post.append(np.argmax(explanation.summary[node_idx].explanation['classprob/post']))

    expl_probs_human.append(explanation.summary[node_idx].explanation['classprob/human'])
    expl_probs_raw.append(explanation.summary[node_idx].explanation['classprob/raw'])
    expl_probs_post.append(explanation.summary[node_idx].explanation['classprob/post'])
    
    
    # complement
    compl_loss_human.append(explanation.summary[node_idx].complement['loss/human'])
    compl_loss_raw.append(explanation.summary[node_idx].complement['loss/raw']) 
    compl_loss_post.append(explanation.summary[node_idx].complement['loss/post']) 

    compl_class_human.append(np.argmax(explanation.summary[node_idx].complement['classprob/human']))
    compl_class_raw.append(np.argmax(explanation.summary[node_idx].complement['classprob/raw']))
    compl_class_post.append(np.argmax(explanation.summary[node_idx].complement['classprob/post']))

    compl_probs_human.append(explanation.summary[node_idx].complement['classprob/human'])
    compl_probs_raw.append(explanation.summary[node_idx].complement['classprob/raw'])
    compl_probs_post.append(explanation.summary[node_idx].complement['classprob/post'])


expl_loss_human = torch.tensor(expl_loss_human)
expl_loss_raw = torch.tensor(expl_loss_raw)
expl_loss_post = torch.tensor(expl_loss_post)

expl_class_human = torch.tensor(expl_class_human)
expl_class_raw = torch.tensor(expl_class_raw)
expl_class_post = torch.tensor(expl_class_post)

expl_probs_human = torch.tensor(expl_probs_human)
expl_probs_raw = torch.tensor(expl_probs_raw)
expl_probs_post = torch.tensor(expl_probs_post)

compl_loss_human = torch.tensor(compl_loss_human)
compl_loss_raw = torch.tensor(compl_loss_raw)
compl_loss_post = torch.tensor(compl_loss_post)

compl_class_human = torch.tensor(compl_class_human)
compl_class_raw = torch.tensor(compl_class_raw)
compl_class_post = torch.tensor(compl_class_post)

compl_probs_human = torch.tensor(compl_probs_human)
compl_probs_raw = torch.tensor(compl_probs_raw)
compl_probs_post = torch.tensor(compl_probs_post)


assert torch.allclose(expl_loss_human, compl_loss_human)
assert torch.allclose(expl_class_human, compl_class_human)
assert torch.allclose(expl_probs_raw, expl_probs_post, rtol=1e-4)
assert torch.allclose(compl_probs_raw, compl_probs_post, rtol=1e-4)

# %%

N = len(class_label)

fidplus_human = torch.abs((y_pred == class_label).to(int) - (compl_class_human == class_label).to(int)).sum() / N
fidminus_human = torch.abs((y_pred == class_label).to(int) - (expl_class_human == class_label).to(int)).sum() / N
fidplus_soft = torch.abs((y_pred == class_label).to(int) - (compl_class_raw == class_label).to(int)).sum() / N
fidminus_soft = torch.abs((y_pred == class_label).to(int) - (expl_class_raw == class_label).to(int)).sum() / N
fidplus_hard = torch.abs((y_pred == class_label).to(int) - (compl_class_post == class_label).to(int)).sum() / N
fidminus_hard = torch.abs((y_pred == class_label).to(int) - (expl_class_post == class_label).to(int)).sum() / N


print("human")
print(f"fidelity+ {fidplus_human:.3f} | fidelity- {fidminus_human:.3f} | loss {expl_loss_human.mean().item():.3f} {compl_loss_human.mean().item():.3f}")

print("raw")
print(f"fidelity+ {fidplus_soft:.3f} | fidelity- {fidminus_soft:.3f} | loss {expl_loss_raw.mean().item():.3f} {compl_loss_raw.mean().item():.3f}")

print("post")
print(f"fidelity+ {fidplus_hard:.3f} | fidelity- {fidminus_hard:.3f} | loss {expl_loss_post.mean().item():.3f} {compl_loss_post.mean().item():.3f}")

# %%

probs_rand = torch.ones_like(probs)
probs_rand = probs_rand / probs_rand.sum(dim=1, keepdim=True)

# %%

# kl div

print("rand", (probs * (probs / probs_rand).log()).mean())
print("human", (probs * (probs / expl_probs_human).log()).mean())
print("expl", (probs * (probs / expl_probs_post).log()).mean())
print("compl", (probs * (probs / compl_probs_post).log()).mean())

# %%

# tv distance

print("rand", (probs - probs_rand).abs().mean())
print("human", (probs - expl_probs_human).abs().mean())
print("expl", (probs - expl_probs_post).abs().mean())
print("compl", (probs - compl_probs_post).abs().mean())
# %%

# cross-entropy, i.e. log score?

print("rand", (probs * probs_rand.log()).mean())
print("human", (probs * expl_probs_human.log()).mean())
print("expl", (probs * expl_probs_post.log()).mean())
print("compl", (probs * compl_probs_post.log()).mean())
# %%
