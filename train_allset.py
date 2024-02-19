# %%

# similar to Allset train.py
# https://github.com/jianhao2016/AllSet/blob/main/src/train.py
# using own training loop


import torch
import torch_sparse
import scipy.sparse as sp

from models.allset import *

from train.train_utils import train_eval_loop
from hgraph.utils import load_hgraph
from torch_geometric.data import Data


class MyArgs:
    def __init__(self):
        # self.dname = "coauthor_cora"
        self.p2raw = "data/AllSet_all_raw_data/coauthorship/"
        self.method = "AllSetTransformer"
        self.add_self_loop = True
        self.exclude_self = False
        self.normtype = "all_one"
        self.runs = 20
        self.train_prop = 0.5
        self.valid_prop = 0.25
        self.LearnMask = True
        self.All_num_layers = 2
        self.dropout = 0.5
        # self.aggregate = "mean"
        self.aggregate = "sum"
        self.normalization = "ln"
        # self.normalization = "None"
        self.deepset_input_norm = True
        self.GPR = True
        # self.MLP_hidden = 64
        self.MLP_hidden = 16
        self.MLP_num_layers = 2
        self.heads = 1
        self.PMA = True
        self.Classifier_hidden = 64
        self.Classifier_hidden = 16
        self.Classifier_num_layers = 2
        self.cuda = 0
        self.lr = 0.001
        self.epochs = 500
        self.HyperGCN_fast = True
        self.HyperGCN_mediators = True
        self.HCHA_symdegnorm = False
        self.output_heads = 1
        self.HNHN_alpha = -1.5
        self.HNHN_beta = -0.5
        self.HNHN_nonlinear_inbetween = True
        self.UniGNN_use_norm = False


args = MyArgs()


# %%

"""
#--------
# Allset dataset loading


dataset = dataset_Hypergraph(
    name=args.dname,
    root = 'data/pyg_data/hypergraph_dataset_updated/',
    p2raw=args.p2raw)

data = dataset.data
print(data)


args.num_features = dataset.num_features
args.num_classes = dataset.num_classes

"""


# %%
#---------

# load your own hgraph as data


def get_weird_edge_index(edge_index, num_nodes):

    weird_edge_index = edge_index.clone()
    weird_edge_index[1,:] = weird_edge_index[1,:] + num_nodes

    weird_edge_index = torch.hstack((weird_edge_index, torch.roll(weird_edge_index, shifts=1, dims=0)))

    return weird_edge_index


# Load example synthetic data hgraph
hgraph = load_hgraph("data/standard/ones1/hgraph1.pickle")

args.num_features = hgraph.x.shape[1]
args.num_classes = hgraph.num_classes

data = Data(
    x=hgraph.x, 
    edge_index=get_weird_edge_index(hgraph.edge_index, hgraph.number_of_nodes()),
    y=hgraph.y,
    n_x=hgraph.number_of_nodes(),
    num_hyperedges=hgraph.number_of_edges(),
    # additionally add your deterministic masks. these are binary masks of shape (num_nodes,)
    train_mask=hgraph.train_mask,
    val_mask=hgraph.val_mask,
    test_mask=hgraph.test_mask,
)

print(data)

#---------

# %%

if not hasattr(data, 'n_x'):
    data.n_x = torch.tensor([data.x.shape[0]])
if not hasattr(data, 'num_hyperedges'):
    # note that we assume the he_id is consecutive.
    data.num_hyperedges = torch.tensor(
        [data.edge_index[0].max()-data.n_x[0]+1])


if args.method in ['AllSetTransformer', 'AllDeepSets']:
    data = ExtractV2E(data)
    if args.add_self_loop:
        data = Add_Self_Loops(data)
    if args.exclude_self:
        data = expand_edge_index(data)

    #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
    # data.norm = torch.ones_like(data.edge_index[0])
    data = norm_contruction(data, option=args.normtype)
elif args.method in ['CEGCN', 'CEGAT']:
    data = ExtractV2E(data)
    data = ConstructV2V(data)
    data = norm_contruction(data, TYPE='V2V')

elif args.method in ['HyperGCN']:
    data = ExtractV2E(data)

elif args.method in ['HNHN']:
    data = ExtractV2E(data)
    if args.add_self_loop:
        data = Add_Self_Loops(data)
    H = ConstructH_HNHN(data)
    data = generate_norm_HNHN(H, data, args)
    data.edge_index[1] -= data.edge_index[1].min()

elif args.method in ['HCHA', 'HGNN']:
    data = ExtractV2E(data)
    if args.add_self_loop:
        data = Add_Self_Loops(data)
#    Make the first he_id to be 0
    data.edge_index[1] -= data.edge_index[1].min()
    
elif args.method in ['UniGCNII']:
    data = ExtractV2E(data)
    if args.add_self_loop:
        data = Add_Self_Loops(data)
    data = ConstructH(data)
    data.edge_index = sp.csr_matrix(data.edge_index)
    # Compute degV and degE
    if args.cuda in [0,1]:
        device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    (row, col), value = torch_sparse.from_scipy(data.edge_index)
    V, E = row, col
    V, E = V.to(device), E.to(device)

    degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
    from torch_scatter import scatter
    degE = scatter(degV[V], E, dim=0, reduce='mean')
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[torch.isinf(degV)] = 1
    args.UniGNN_degV = degV
    args.UniGNN_degE = degE

    V, E = V.cpu(), E.cpu()
    del V
    del E

#     Get splits
split_idx_lst = []
for run in range(args.runs):
    split_idx = rand_train_test_idx(
        data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
    split_idx_lst.append(split_idx)




def parse_method(args, data):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HCHA(args, use_attention=False)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args, use_attention=True)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII':
            if args.cuda in [0,1]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)
            model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                             V=V, E=E)
    #     Below we can add different model, such as HyperGCN and so on
    return model



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = parse_method(args, data)
# put things to device
if args.cuda in [0, 1]:
    device = torch.device('cuda:'+str(args.cuda)
                            if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')



model, data = model.to(device), data.to(device)
if args.method == 'UniGCNII':
    args.UniGNN_degV = args.UniGNN_degV.to(device)
    args.UniGNN_degE = args.UniGNN_degE.to(device)

num_params = count_parameters(model)

print(f"{num_params=}")
print(model)


# %%


# Train, using run 0 of the split


run = 0
split_idx = split_idx_lst[run]
train_mask = split_idx['train'].to(device)
val_mask = split_idx['valid'].to(device)
test_mask = split_idx['test'].to(device)

model.reset_parameters()



train_stats, best_model = train_eval_loop(
    model=model,
    hgraph=data,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    # train_mask=data.train_mask,
    # val_mask=data.val_mask,
    # test_mask=data.test_mask,
    lr=args.lr,
    num_epochs=args.epochs,
    contr_lambda=0.0,
    printevery=50,
)



# %%