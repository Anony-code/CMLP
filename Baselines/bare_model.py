import torch
import dgl
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from Backbones.utils import compute_energy

class NET(torch.nn.Module):

    """
    Bare model baseline for NCGL tasks

    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

    """

    def __init__(self,
                 model,
                 task_manager,
                 args):
        """
        The initialization of the baseline

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.
        """
        super(NET, self).__init__()

        self.task_manager = task_manager

        # backbone model
        self.net = model[0]
        self.mlp = model[1]
        self.gate = model[2]
        self.gmm = model[3]   # list of gmm models
        self.flow = model[4]

        # # # setup optimizer
        # filtered_params = [
        #     name for name, param in list(self.mlp.named_parameters())]
        # print(filtered_params)
        # exit()


        self.update_para = [param for name, param in list(self.mlp.named_parameters()) 
                            if name in ['mlp_layers.0.weight.0', 'mlp_layers.0.bias.0', 'mlp_layers.1.bias', 'mlp_layers.1.weight.0', 'mlp_layers.1.clf.0']]
        # self.opt = torch.optim.Adam(list(self.net.parameters())+list(self.mlp.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        self.opt = torch.optim.Adam(list(self.net.parameters()) + self.update_para, lr=args.lr, weight_decay=args.weight_decay)
        # self.opt_mlp = torch.optim.Adam(self.mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # self.opt = torch.optim.SGD(list(self.net.parameters()) + self.update_para, 
        #                    lr=args.lr, 
        #                    weight_decay=args.weight_decay, 
        #                    momentum=0.9)  # Use momentum to stabilize updates

        # setup loss
        self.ce = torch.nn.functional.cross_entropy

        self.opt_gate = torch.optim.Adam(self.gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the class-IL setting.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
        :param dataset: The entire dataset (not in use in the current baseline).

        """
        self.net.train()
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output = self.net(g, features)
        if isinstance(output,tuple):
            output = output[0]
        output_labels = labels[train_ids]
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        if args.classifier_increase:
            loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], output_labels, weight=loss_w_)
        loss.backward()
        self.opt.step()

    def observe_task_IL(self, args, g, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                The method for learning the given tasks under the task-IL setting.

                :param args: Same as the args in __init__().
                :param g: The graph of the current task.
                :param features: Node features of the current task.
                :param labels: Labels of the nodes in the current task.
                :param t: Index of the current task.
                :param train_ids: The indices of the nodes participating in the training.
                :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                :param dataset: The entire dataset (not in use in the current baseline).

                """
        self.net.train()
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output = self.net(g, features)
        if isinstance(output, tuple):
            output = output[0]
        output_labels = labels[train_ids]
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        output_labels = output_labels-offset1
        loss = self.ce(output[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        loss.backward()
        self.opt.step()

    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                The method for learning the given tasks under the task-IL setting with mini-batch training.

                :param args: Same as the args in __init__().
                :param g: The graph of the current task.
                :param dataloader: The data loader for mini-batch training
                :param features: Node features of the current task.
                :param labels: Labels of the nodes in the current task.
                :param t: Index of the current task.
                :param train_ids: The indices of the nodes participating in the training.
                :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                :param dataset: The entire dataset (currently not in use).

                """
        # now compute the grad on the current task
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_labels = output_labels - offset1
            output_predictions, _ = self.net.forward_batch(blocks, input_features)
            loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            loss.backward()
            self.opt.step()

    def observe_class_IL_last_batch(self, args,dataloader, t):
        """
                        The method for learning the given tasks under the class-IL setting with mini-batch training.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param dataloader: The data loader for mini-batch training.
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                        :param dataset: The entire dataset (currently not in use).

                        """
        # offset1, offset2 = self.task_manager.get_label_offset(t)
        offset1 = args.n_cls_per_task
        offset2 = args.n_cls_per_task*5

        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            self.mlp.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']

            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_features = blocks[-1].dstdata['feat']

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_predictions, _ = self.net.forward_batch(blocks, input_features)
            # outout_mlp, _ = self.mlp.forward_batch(blocks, output_features, 1)
            # output_gate = self.gate.forward_batch(blocks, output_features)
            # if args.classifier_increase:
            #     loss = self.ce(output_predictions[:, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])
            # else:
            loss = self.ce(output_predictions, output_labels, weight=loss_w_)

            # print(loss)

            loss.backward()
            self.opt.step()

    

    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the class-IL setting with mini-batch training.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param dataloader: The data loader for mini-batch training.
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                        :param dataset: The entire dataset (currently not in use).

                        """
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            self.mlp.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']

            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_features = blocks[-1].dstdata['feat']

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            output_predictions, _ = self.net.forward_batch(blocks, input_features)
            outout_mlp, _ = self.mlp.forward_batch(blocks, output_features, 1)
            # output_gate = self.gate.forward_batch(blocks, output_features)




            if args.classifier_increase:
                loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output_predictions, output_labels, weight=loss_w_)




            # print("GNN:", loss)

            # loss.backward(retain_graph=True)
            # self.opt.step()
            # for param in self.net.parameters():
            #     param.requires_grad = False

            ######################
            T = 2.0
            # distill_loss = F.kl_div(
            #     F.log_softmax(output_predictions[:, offset1:offset2] / T, dim=1), 
            #     F.softmax(outout_mlp / T, dim=1).detach(),  # Prevents backprop into teacher
            #     reduction='batchmean'
            # )* (T ** 2)
            distill_loss = F.kl_div(F.log_softmax(output_predictions[:, offset1:offset2] / T, dim=1), F.softmax(outout_mlp / T, dim=1), reduction='batchmean') * (T ** 2)
            
            mlp_ce = self.ce(outout_mlp[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            loss = distill_loss + mlp_ce + loss
            # loss = mlp_ce

            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            loss.backward()
            self.opt.step()

            # self.gmm.zero_grad()
            # output_label_gate = torch.full(
            #     (output_gate.size(0),),  # Shape = [batch_size]
            #     fill_value=0,
            #     dtype=torch.long,
            #     device=output_gate.device
            # )
            # loss_gate = self.ce(output_gate[:, offset1:offset2], output_label_gate)

            # loss_gate.backward()
            # self.opt_gate.step()

            # self.gmm[0].forward_batch(blocks, output_features)

    def observe_class_IL_batch_energy(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        self.mlp.eval()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        ene = []
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            output_features = blocks[-1].dstdata['feat']
            outout_mlp, _ = self.mlp.forward_batch(blocks, output_features, 1)
            energies = compute_energy(outout_mlp, T=1.0)
            ene.append(energies)
        ene = torch.cat(ene, dim=0)
        print("training:", ene.mean())



    def observe_class_IL_batch_flow(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset):
        """
                        The method for learning the given tasks under the class-IL setting with mini-batch training.

                        :param args: Same as the args in __init__().
                        :param g: The graph of the current task.
                        :param dataloader: The data loader for mini-batch training.
                        :param features: Node features of the current task.
                        :param labels: Labels of the nodes in the current task.
                        :param t: Index of the current task.
                        :param train_ids: The indices of the nodes participating in the training.
                        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
                        :param dataset: The entire dataset (currently not in use).

                        """
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']

            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_features = blocks[-1].dstdata['feat']

            self.flow[0].fit_pca(output_features)
            flow_data = TensorDataset(output_features)
            flow_dataloader = DataLoader(flow_data, batch_size=1024, shuffle=True)

            # Train Normalizing Flow
            self.flow[0].fit_flow(flow_dataloader, epochs=350, lr=1e-3)

        
            density_value = self.flow[0].compute_density(output_features)
            proto = self.flow[0].find_prototypes(output_features, output_labels, density_value)
            print(proto)




    def observe_class_IL_batch_MLP(self, args, g, dataloader, features, labels, t, train_ids, ids_per_cls, dataset, step):
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for input_nodes, output_nodes, blocks in dataloader:
            self.mlp.zero_grad()

            self.update_para = [param for name, param in list(self.mlp.named_parameters()) 
                            if name in ['mlp_layers.0.weight.'+str(step), 'mlp_layers.0.bias.'+str(step), 'mlp_layers.1.bias', 'mlp_layers.1.weight.'+str(step), 'mlp_layers.1.clf.'+str(step)]]
            self.opt = torch.optim.Adam(list(self.net.parameters()) + self.update_para, lr=args.lr, weight_decay=args.weight_decay)

            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_features = blocks[-1].dstdata['feat']

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]

            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            outout_mlp, _ = self.mlp.forward_batch(blocks, output_features, step+1)

            # output_gate = self.gate.forward_batch(blocks, output_features)
            # mlp_ce = self.ce(outout_mlp[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
            mlp_ce = self.ce(outout_mlp, output_labels-args.n_cls_per_task*step, weight=loss_w_[offset1+args.n_cls_per_task*step: offset2])
            loss =  mlp_ce
        
            loss.backward()
            self.opt.step()


            # self.gate.zero_grad()
            # output_label_gate = torch.full(
            #     (output_gate.size(0),),  # Shape = [batch_size]
            #     fill_value=step,
            #     dtype=torch.long,
            #     device=output_gate.device
            # )
            # loss_gate = self.ce(output_gate[:, offset1:offset2], output_label_gate)

            # loss_gate.backward()
            # self.opt_gate.step()

            # self.gmm[step].forward_batch(blocks, output_features)



