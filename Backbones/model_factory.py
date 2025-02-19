import torch.nn.functional as F
from .gnns import GAT, GCN, GIN, MLP, Gate, GMM, NormalizingFlowWithPCA

def get_model(dataset, args):
    n_classes = args.n_cls_per_task
    print('n_classes', n_classes)
    if args.backbone == 'GAT':
        heads = ([args.GAT_args['heads']] * args.GAT_args['num_layers']) + [args.GAT_args['out_heads']]
        model = GAT(args, heads, F.elu)
    elif args.backbone == 'GCN':
        model = GCN(args)
    elif args.backbone == 'GIN':
        model = GIN(args)
    return model

def get_mlp_model(dataset, args):
    return MLP(args)

def get_gate_model(dataset, args):
    return Gate(args)

def get_gmm_model(dataset, args):
    return GMM(args)

def get_flow_model(dataset, args):
    return NormalizingFlowWithPCA(args)