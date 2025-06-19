import random
import pickle
import numpy as np
import torch
from torch import Tensor, device, dtype
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.data import CoraGraphDataset, CoraFullDataset, register_data_args, RedditDataset, AmazonCoBuyComputerDataset
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
import copy
from sklearn.metrics import roc_auc_score, average_precision_score

class Linear_IL(nn.Linear):
    def forward(self, input: Tensor, n_cls=10000, normalize = True) -> Tensor:
        if normalize:
            return F.linear(F.normalize(input,dim=-1), F.normalize(self.weight[0:n_cls],dim=-1), bias=None)
        else:
            return F.linear(input, self.weight[0:n_cls], bias=None)

def accuracy(logits, labels, new_ids_per_cls, cls_balance=True, ids_per_cls=None):
    if cls_balance:
        logi = logits.cpu().numpy()
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()

        # acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        acc_per_cls = [torch.sum((indices == labels)[new_ids_per_cls[id]]) / len(ids_per_cls[id]) for id in range(len(ids_per_cls))]

        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def mean_AP(args,logits, labels, cls_balance=True, ids_per_cls=None):
    eval_ogb = Evaluator(args.dataset)
    pos = (F.sigmoid(logits)>0.5)
    APs = 0
    if cls_balance:
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        input_dict = {"y_true": labels, "y_pred": logits}

        eval_result_ogb = eval_ogb.eval(input_dict)
        for c,ids in enumerate(ids_per_cls):
            TP_ = (pos[ids,c]*labels[ids,c]).sum()
            FP_ = (pos[ids,c]*(labels[ids, c]==False)).sum()
            med0 = TP_ + FP_ + 0.0001
            med1 = TP_ / med0
            APs += med1
        med2 = APs/labels.shape[1]

            #mAP_per_cls.append((TP / (TP+FP)).mean().item())
        #return (TP / (TP+FP)).mean().item()

        return med2.item()
    
def compute_energy(logits, T=1.0):
    # logits: [batch_size, num_classes]
    # 计算能量值: E(x) = -T * log(sum(exp(logits / T)))
    energy = -T * torch.logsumexp(logits / T, dim=1)
    return energy


def evaluate_GNN_batch(args, model, g, labels, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None):
    model.eval()
    with torch.no_grad():
        dataloader = dgl.dataloading.DataLoader(g.cpu(), list(range(labels.shape[0])), args.nb_sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
        output = torch.tensor([]).cuda(args.gpu)
        output_l = torch.tensor([]).cuda(args.gpu)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_predictions, _ = model.forward_batch(blocks, input_features)
            output = torch.cat((output,output_predictions),dim=0)
            output_l = torch.cat((output_l, output_labels), dim=0)

        #output, _ = model(g, features)
        #judget = (labels==output_l).sum()
        logits = output
        new_ids_per_cls = ids_per_cls
        
        return accuracy(logits, labels.cuda(args.gpu), new_ids_per_cls, cls_balance=cls_balance, ids_per_cls=ids_per_cls)



def evaluate_batch(args, model, gate_model, gmm_model, g, features, labels, mask, label_offset1, label_offset2, task, step, gnn=False, cls_balance=True, ids_per_cls=None):
    model.eval()
    gate_model.eval()
    with torch.no_grad():
        # print("++++++++++++++++++++++\n")
        dataloader = dgl.dataloading.DataLoader(g.cpu(), list(range(labels.shape[0])), args.nb_sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
        output_0 = torch.tensor([]).cuda(args.gpu)
        output_1 = torch.tensor([]).cuda(args.gpu)
        output_2 = torch.tensor([]).cuda(args.gpu)
        output_3 = torch.tensor([]).cuda(args.gpu)
        output_4 = torch.tensor([]).cuda(args.gpu)
        output_l = torch.tensor([]).cuda(args.gpu)
        expert_mix = torch.tensor([]).cuda(args.gpu)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_features = blocks[-1].dstdata['feat']

            # expert = gate_model.forward_batch(blocks, output_features)
            # expert_mix = torch.cat((expert_mix, expert), dim=0)
            # _, expert_indices = torch.max(expert, dim=1)
            # expert_value = torch.mode(expert_indices).values
            # # print(expert_value)

            # gmm_pca = gmm_model.pca_(output_features)
            # dot_similarity = torch.from_numpy(np.dot(gmm_pca, protos.T))

            # expert_mix = torch.cat((expert_mix, dot_similarity.to(device='cuda:{}'.format(args.gpu))), dim=0)
            

            # ###########################
            # output_predictions_0, _ = model.forward_batch(blocks, output_features, task+1)
            # output_0 = torch.cat((output_0,output_predictions_0),dim=0)
            # logits = output_0
            # new_ids_per_cls = ids_per_cls




            ###########################################################
            if task == 0:
                if gnn == False:
                    output_predictions_0, _ = model.forward_batch(blocks, output_features, task+1)
                    output_0 = torch.cat((output_0,output_predictions_0),dim=0)
                if gnn == True:
                    output_predictions_0, _ = model.forward_batch(blocks, input_features)
                    output_0 = torch.cat((output_0,output_predictions_0),dim=0)
            if task == 1:
                output_predictions_1, _ = model.forward_batch(blocks, output_features, task+1)
                output_predictions_0, _ = model.forward_batch(blocks, output_features, task)
                output_1 = torch.cat((output_1,output_predictions_1),dim=0)
                output_0 = torch.cat((output_0,output_predictions_0),dim=0)
            if task == 2:
                output_predictions_2, _ = model.forward_batch(blocks, output_features, task+1)
                output_predictions_1, _ = model.forward_batch(blocks, output_features, task)
                output_predictions_0, _ = model.forward_batch(blocks, output_features, task-1)
                output_2 = torch.cat((output_2,output_predictions_2),dim=0)
                output_1 = torch.cat((output_1,output_predictions_1),dim=0)
                output_0 = torch.cat((output_0,output_predictions_0),dim=0)
            if task == 3:
                output_predictions_3, _ = model.forward_batch(blocks, output_features, task+1)
                output_predictions_2, _ = model.forward_batch(blocks, output_features, task)
                output_predictions_1, _ = model.forward_batch(blocks, output_features, task-1)
                output_predictions_0, _ = model.forward_batch(blocks, output_features, task-2)
                output_3 = torch.cat((output_3,output_predictions_3),dim=0)
                output_2 = torch.cat((output_2,output_predictions_2),dim=0)
                output_1 = torch.cat((output_1,output_predictions_1),dim=0)
                output_0 = torch.cat((output_0,output_predictions_0),dim=0)
            if task == 4:
                output_predictions_4, _ = model.forward_batch(blocks, output_features, task+1)
                output_predictions_3, _ = model.forward_batch(blocks, output_features, task)
                output_predictions_2, _ = model.forward_batch(blocks, output_features, task-1)
                output_predictions_1, _ = model.forward_batch(blocks, output_features, task-2)
                output_predictions_0, _ = model.forward_batch(blocks, output_features, task-3)
                output_4 = torch.cat((output_4,output_predictions_4),dim=0)
                output_3 = torch.cat((output_3,output_predictions_3),dim=0)
                output_2 = torch.cat((output_2,output_predictions_2),dim=0)
                output_1 = torch.cat((output_1,output_predictions_1),dim=0)
                output_0 = torch.cat((output_0,output_predictions_0),dim=0)
            
            output_l = torch.cat((output_l, output_labels), dim=0)

        #output, _ = model(g, features)
        #judget = (labels==output_l).sum()
        # label_offset2 = 28
        # label_offset1 = 14

        # for input_nodes, output_nodes, blocks in dataloader:
        #     blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
        #     input_features = blocks[0].srcdata['feat']
        #     output_labels = blocks[-1].dstdata['label'].squeeze()
        #     output_features = blocks[-1].dstdata['feat']
        #     # output_predictions, _ = model.forward_batch(blocks, input_features)
        #     output_predictions, _ = model.forward_batch(blocks, output_features, task+1)
        #     output = torch.cat((output,output_predictions),dim=0)
        #     output_l = torch.cat((output_l, output_labels), dim=0)
        

        if task == 0 and gnn == False:
            logits = output_0
            new_ids_per_cls = ids_per_cls
        if task == 0 and gnn == True:
            logits = output_0[:, 0:args.n_cls_per_task]
            new_ids_per_cls = ids_per_cls

        if task == 1:
            logits_0 = output_0
            logits_1 = output_1
            lg = [logits_0, logits_1]
            # print(labels.tolist())
            labels = labels -step*args.n_cls_per_task
            # labels = labels


            new_ids_per_cls = []
            for sublist in ids_per_cls:
                o0 = output_0[torch.tensor(sublist)]
                o1 = output_1[torch.tensor(sublist)]
                ene_0 = compute_energy(o0, T=1.0)
                ene_1 = compute_energy(o1, T=1.0)
                min_values, min_indices = torch.min(torch.stack((ene_0, ene_1)), dim=0)
                zero_indices = torch.nonzero(min_indices ==step, as_tuple=True)[0]
                sublist_zero = [sublist[i] for i in zero_indices.tolist()]
                new_ids_per_cls.append(sublist_zero)
                
            logits = lg[step]

        if task == 2:
            logits_0 = output_0
            logits_1 = output_1
            logits_2 = output_2
            lg = [logits_0, logits_1, logits_2]
            # print(labels.tolist())
            labels = labels -step*args.n_cls_per_task
            # labels = labels

            new_ids_per_cls = []
            for sublist in ids_per_cls:
                o0 = output_0[torch.tensor(sublist)]
                o1 = output_1[torch.tensor(sublist)]
                o2 = output_2[torch.tensor(sublist)]
                ene_0 = compute_energy(o0, T=1.0)
                ene_1 = compute_energy(o1, T=1.0)
                ene_2 = compute_energy(o2, T=1.0)
                min_values, min_indices = torch.min(torch.stack((ene_0, ene_1, ene_2)), dim=0)
                zero_indices = torch.nonzero(min_indices ==step, as_tuple=True)[0]
                sublist_zero = [sublist[i] for i in zero_indices.tolist()]
                new_ids_per_cls.append(sublist_zero)
                
            logits = lg[step]

        if task == 3:
            logits_0 = output_0
            logits_1 = output_1
            logits_2 = output_2
            logits_3 = output_3
            lg = [logits_0, logits_1, logits_2, logits_3]
            # print(labels.tolist())
            labels = labels -step*args.n_cls_per_task
            # labels = labels

            new_ids_per_cls = []
            for sublist in ids_per_cls:
                o0 = output_0[torch.tensor(sublist)]
                o1 = output_1[torch.tensor(sublist)]
                o2 = output_2[torch.tensor(sublist)]
                o3 = output_3[torch.tensor(sublist)]
                ene_0 = compute_energy(o0, T=1.0)
                ene_1 = compute_energy(o1, T=1.0)
                ene_2 = compute_energy(o2, T=1.0)
                ene_3 = compute_energy(o3, T=1.0)
                min_values, min_indices = torch.min(torch.stack((ene_0, ene_1, ene_2, ene_3)), dim=0)
                zero_indices = torch.nonzero(min_indices ==step, as_tuple=True)[0]
                sublist_zero = [sublist[i] for i in zero_indices.tolist()]
                new_ids_per_cls.append(sublist_zero)
                
            logits = lg[step]
        

        if task == 4:
            logits_0 = output_0
            logits_1 = output_1
            logits_2 = output_2
            logits_3 = output_3
            logits_4 = output_4
            lg = [logits_0, logits_1, logits_2, logits_3, logits_4]
            # print(labels.tolist())
            labels = labels -step*args.n_cls_per_task
            # labels = labels

            new_ids_per_cls = []
            for sublist in ids_per_cls:
                o0 = output_0[torch.tensor(sublist)]
                o1 = output_1[torch.tensor(sublist)]
                o2 = output_2[torch.tensor(sublist)]
                o3 = output_3[torch.tensor(sublist)]
                o4 = output_4[torch.tensor(sublist)]
                ene_0 = compute_energy(o0, T=1.0)
                ene_1 = compute_energy(o1, T=1.0)
                ene_2 = compute_energy(o2, T=1.0)
                ene_3 = compute_energy(o3, T=1.0)
                ene_4 = compute_energy(o4, T=1.0)
                min_values, min_indices = torch.min(torch.stack((ene_0, ene_1, ene_2, ene_3, ene_4)), dim=0)
                zero_indices = torch.nonzero(min_indices ==step, as_tuple=True)[0]
                sublist_zero = [sublist[i] for i in zero_indices.tolist()]
                new_ids_per_cls.append(sublist_zero)
                
            logits = lg[step]

        #############################################################


        # print("task:", task)
        # if task == 0:
        #     logits_0 = output
        #     energies_0 = compute_energy(logits_0, T=1.0)
        #     print("energies-0:", energies_0.min())
        #     print("energies-0:", energies_0.max())
        #     print("energies-0:", energies_0.mean())

        # if task == 1:
        #     logits_10 = output
        #     energies_10 = compute_energy(logits_10, T=1.0)
        #     print("energies-0:", energies_10.min())
        #     print("energies-0:", energies_10.max())
        #     print("energies-0:", energies_10.mean())

        #     # logits_11 = output
        #     # energies_11 = compute_energy(logits_11, T=1.0)
        #     # print("energies-1:", energies_11.min())
        #     # print("energies-1:", energies_11.max())
        #     # print("energies-1:", energies_11.mean())
        
        # if task == 2:
        #     logits_20 = output[:, 0:14]
        #     energies_20 = compute_energy(logits_20, T=1.0)
        #     print("energies-0:", energies_20.min())
        #     print("energies-0:", energies_20.max())
        #     print("energies-0:", energies_20.mean())

        #     logits_21 = output[:, 14:28]
        #     energies_21 = compute_energy(logits_21, T=1.0)
        #     print("energies-1:", energies_21.min())
        #     print("energies-1:", energies_21.max())
        #     print("energies-1:", energies_21.mean())

        #     logits_22 = output[:, 28:42]
        #     energies_22 = compute_energy(logits_22, T=1.0)
        #     print("energies-2:", energies_22.min())
        #     print("energies-2:", energies_22.max())
        #     print("energies-2:", energies_22.mean())
        
        # if task == 3:
        #     logits_30 = output[:, 0:14]
        #     energies_30 = compute_energy(logits_30, T=1.0)
        #     print("energies-0:", energies_30.min())
        #     print("energies-0:", energies_30.max())
        #     print("energies-0:", energies_30.mean())

        #     logits_31 = output[:, 14:28]
        #     energies_31 = compute_energy(logits_31, T=1.0)
        #     print("energies-1:", energies_31.min())
        #     print("energies-1:", energies_31.max())
        #     print("energies-1:", energies_31.mean())

        #     logits_32 = output[:, 28:42]
        #     energies_32 = compute_energy(logits_32, T=1.0)
        #     print("energies-2:", energies_32.min())
        #     print("energies-2:", energies_32.max())
        #     print("energies-2:", energies_32.mean())

        #     logits_33 = output[:, 42:56]
        #     energies_33 = compute_energy(logits_33, T=1.0)
        #     print("energies-3:", energies_33.min())
        #     print("energies-3:", energies_33.max())
        #     print("energies-3:", energies_33.mean())

    

        # _, indices_expert = torch.max(expert_mix, dim=1)
        # expert_current = []
        # expert_current.extend(indices_expert[ids] for ids in ids_per_cls)
        # expert_current = torch.cat(expert_current)
        # expert_value = expert_current.float().mean()
        # print(expert_current)

        # range_1 = (0, 13)
        # range_2 = (14, 27)
        # range_3 = (28, 41)
        # range_4 = (42, 55)
        # range_5 = (56, 69)

        # # 统计每个范围内的数量
        # count_range_1 = ((expert_current >= range_1[0]) & (expert_current <= range_1[1])).sum().item()
        # count_range_2 = ((expert_current >= range_2[0]) & (expert_current <= range_2[1])).sum().item()
        # count_range_3 = ((expert_current >= range_3[0]) & (expert_current <= range_3[1])).sum().item()
        # count_range_4 = ((expert_current >= range_4[0]) & (expert_current <= range_4[1])).sum().item()
        # count_range_5 = ((expert_current >= range_5[0]) & (expert_current <= range_5[1])).sum().item()

        # print(f"0-13: {count_range_1}, 14-27: {count_range_2}, 28-41: {count_range_3}, 42-55: {count_range_4}, 56-69: {count_range_5}")


        # print(expert_value)

        # if expert_value <= 14:
        #     logits = output[:, 0:14]
        # elif expert_value <= 28 and expert_value > 14:
        #     logits = output[:, 14:28]
        # elif expert_value <= 42 and expert_value > 28:
        #     logits = output[:, 28:42]
        # elif expert_value <= 56 and expert_value > 42:
        #     logits = output[:, 42:56]
        # else:
        #     logits = output[:, 56:70]

        if cls_balance:
            return accuracy(logits, labels.cuda(args.gpu), new_ids_per_cls, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask].cuda(args.gpu), new_ids_per_cls, cls_balance=cls_balance, ids_per_cls=ids_per_cls)

        

def evaluate(model, g, features, labels, mask, label_offset1, label_offset2, cls_balance=True, ids_per_cls=None, save_logits_name=None):
    model.eval()
    with torch.no_grad():
        output, _ = model(g, features)
        logits = output[:, label_offset1:label_offset2]
        if save_logits_name is not None:
            with open(
                    '/store/continual_graph_learning/baselines_by_TWP/NCGL/results/logits_for_tsne/{}.pkl'.format(
                        save_logits_name), 'wb') as f:
                pickle.dump({'logits':logits,'ids_per_cls':ids_per_cls}, f)

        if cls_balance:
            return accuracy(logits, labels, cls_balance=cls_balance, ids_per_cls=ids_per_cls)
        else:
            return accuracy(logits[mask], labels[mask], cls_balance=cls_balance, ids_per_cls=ids_per_cls)

class incremental_graph_trans_(nn.Module):
    def __init__(self, dataset, n_cls):
        super().__init__()
        # transductive setting
        self.graph, self.labels = dataset[0]
        #self.graph = dgl.add_reverse_edges(self.graph)
        #self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['label'] = self.labels
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_cls = n_cls
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_nodes = self.labels.shape[0]
        self.tr_va_te_split = dataset[1]

    def get_graph(self, tasks_to_retain=[], node_ids = None, remove_edges=True):
        # get the partial graph
        # tasks-to-retain: classes retained in the partial graph
        # tasks-to-infer: classes to predict on the partial graph
        node_ids_ = copy.deepcopy(node_ids)
        node_ids_retained = []
        ids_train_old, ids_valid_old, ids_test_old = [], [], []
        if len(tasks_to_retain) > 0:
            # retain nodes according to classes
            for t in tasks_to_retain:
                ids_train_old.extend(self.tr_va_te_split[t][0])
                ids_valid_old.extend(self.tr_va_te_split[t][1])
                ids_test_old.extend(self.tr_va_te_split[t][2])
                node_ids_retained.extend(self.tr_va_te_split[t][0] + self.tr_va_te_split[t][1] + self.tr_va_te_split[t][2])
            subgraph_0 = dgl.node_subgraph(self.graph, node_ids_retained, store_ids=True)
            if node_ids_ is None:
                subgraph = subgraph_0
        if node_ids_ is not None:
            # retrain the given nodes
            if not isinstance(node_ids_[0],list):
                # if nodes are not divided into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_, store_ids=True)
                if remove_edges:
                    # to facilitate the methods like ER-GNN to only retrieve nodes
                    n_edges = subgraph_1.edges()[0].shape[0]
                    subgraph_1.remove_edges(list(range(n_edges)))
            elif isinstance(node_ids_[0],list):
                # if nodes are diveded into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_[0], store_ids=True) # load the subgraph containing nodes of the first task
                node_ids_.pop(0)
                for ids in node_ids_:
                    # merge the remaining nodes
                    subgraph_1 = dgl.batch([subgraph_1,dgl.node_subgraph(self.graph, ids, store_ids=True)])

            if len(tasks_to_retain)==0:
                subgraph = subgraph_1

        if len(tasks_to_retain)>0 and node_ids is not None:
            subgraph = dgl.batch([subgraph_0,subgraph_1])

        old_ids = subgraph.ndata['_ID'].cpu()
        ids_train = [(old_ids == i).nonzero()[0][0].item() for i in ids_train_old]
        ids_val = [(old_ids == i).nonzero()[0][0].item() for i in ids_valid_old]
        ids_test = [(old_ids == i).nonzero()[0][0].item() for i in ids_test_old]
        node_ids_per_task_reordered = []
        for c in tasks_to_retain:
            ids = (subgraph.ndata['label'] == c).nonzero()[:, 0].view(-1).tolist()
            node_ids_per_task_reordered.append(ids)
        subgraph = dgl.add_self_loop(subgraph)

        return subgraph, node_ids_per_task_reordered, [ids_train, ids_val, ids_test]

def train_valid_test_split(ids,ratio_valid_test):
    va_te_ratio = sum(ratio_valid_test)
    train_ids, va_te_ids = train_test_split(ids, test_size=va_te_ratio)
    return [train_ids] + train_test_split(va_te_ids, test_size=ratio_valid_test[1]/va_te_ratio)


class NodeLevelDataset(incremental_graph_trans_):
    def __init__(self,name='ogbn-arxiv',IL='class',default_split=False,ratio_valid_test=None,args=None):
        r""""
        name: name of the dataset
        IL: use task- or class-incremental setting
        default_split: if True, each class is split according to the splitting of the original dataset, which may cause the train-val-test ratio of different classes greatly different
        ratio_valid_test: in form of [r_val,r_test] ratio of validation and test set, train set ratio is directly calculated by 1-r_val-r_test
        """

        # return an incremental graph instance that can return required subgraph upon request
        if name[0:4] == 'ogbn':
            data = DglNodePropPredDataset(name, root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name in ['CoraFullDataset', 'CoraFull','corafull', 'CoraFull-CL','Corafull-CL']:
            data = CoraFullDataset()
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
        elif name in ['AmazonCoBuyComputerDataset', 'Amazon','amazon', 'amazon-CL','Amazon-CL']:
            data = AmazonCoBuyComputerDataset()
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
        elif name in ['reddit','Reddit','Reddit-CL']:
            data = RedditDataset(self_loop=False)
            graph = data[0]
            label = graph.ndata['label'].view(-1, 1)
        elif name == 'Arxiv-CL':
            data = DglNodePropPredDataset('ogbn-arxiv', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name == 'Products-CL':
            data = DglNodePropPredDataset('ogbn-products', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        else:
            print('invalid data name')
        n_cls = data.num_classes
        cls = [i for i in range(n_cls)]
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        cls_sizes = {c: len(cls_id_map[c]) for c in cls_id_map}
        for c in cls_sizes:
            if cls_sizes[c] < 2:
                cls.remove(c) # remove classes with less than 2 examples, which cannot be split into train, val, test sets
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        n_cls = len(cls)
        if default_split:
            split_idx = data.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"].tolist(), split_idx["valid"].tolist(), split_idx[
                "test"].tolist()
            tr_va_te_split = {c: [list(set(cls_id_map[c]).intersection(set(train_idx))),
                                  list(set(cls_id_map[c]).intersection(set(valid_idx))),
                                  list(set(cls_id_map[c]).intersection(set(test_idx)))] for c in cls}

        elif not default_split:
            split_name = f'{args.data_path}/tr{round(1-ratio_valid_test[0]-ratio_valid_test[1],2)}_va{ratio_valid_test[0]}_te{ratio_valid_test[1]}_split_{name}.pkl'
            try:
                tr_va_te_split = pickle.load(open(split_name, 'rb')) # could use same split across different experiments for consistency
            except:
                if ratio_valid_test[1] > 0:
                    tr_va_te_split = {c: train_valid_test_split(cls_id_map[c], ratio_valid_test=ratio_valid_test)
                                      for c in
                                      cls}
                    print(f'splitting is {ratio_valid_test}')
                elif ratio_valid_test[1] == 0:
                    tr_va_te_split = {c: [cls_id_map[c], [], []] for c in
                                      cls}
                with open(split_name, 'wb') as f:
                    pickle.dump(tr_va_te_split, f)
        super().__init__([[graph, label], tr_va_te_split], n_cls)