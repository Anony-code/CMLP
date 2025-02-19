from .gnnconv import GATConv, GCNLayer, GINConv
from .layers import PairNorm
from .utils import *
import math
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import torch.nn as nn
from nflows.distributions import StandardNormal
from nflows.transforms import AffineCouplingTransform, ReversePermutation, CompositeTransform
from nflows.flows import Flow
from nflows.utils import create_alternating_binary_mask
from nflows.nn import nets
import torch.optim as optim
from nflows.transforms import MaskedAffineAutoregressiveTransform
from nflows.transforms import LULinear


linear_choices = {'nn.Linear':nn.Linear, 'Linear_IL':Linear_IL}

class AdaptiveLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(AdaptiveLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.step = 5

        # self.weight = nn.ParameterList([torch.Tensor(out_features, in_features)) for i in range(self.step)]
        # self.bias = [nn.Parameter(torch.Tensor(out_features)) for i in range(self.step)]
        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(out_features, in_features)) for _ in range(self.step)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(out_features)) for _ in range(self.step)]) if bias else None
        # self.weight = nn.Parameter(torch.Tensor(out_features*self.step, in_features))
        # self.bias = nn.Parameter(torch.Tensor(out_features*self.step)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and bias
        for i in range(self.step):
            nn.init.xavier_uniform_(self.weight[i])
            if self.bias[i] is not None:
                nn.init.zeros_(self.bias[i])

    def forward(self, input, step):

        # weight = self.weight[step-1]
        # bias = self.bias[step-1]
        # return torch.matmul(input, weight.T.to(input.device)) + bias.to(input.device)

        ############################### progressive manner
        weight = self.weight[0]
        bias = self.bias[0]
        
        if step > 1:
            for i in range(1, step):
                weight = torch.cat((weight, self.weight[i]), dim=0)
                bias = torch.cat((bias, self.bias[i]), dim=0)
        return torch.matmul(input, weight.T.to(input.device)) + bias.to(input.device)

        # return torch.matmul(input, self.weight[:step*self.out_features, :].T) + self.bias[:step*self.out_features]
        ##################################
        
    def forward_test(self, input, step):
        weight = self.weight[0]
        bias = self.bias[0]
        if step > 1:
            for i in range(1, step):
                weight = weight + self.weight[i]
                bias = bias + self.bias[i]
        return torch.matmul(input, weight.T.to(input.device)) + bias.to(input.device)


    
    # def continual(self):
    #     new_weight = torch.randn(self.out_features, self.in_features, device=self.weight.device)
    #     updated_weight = torch.cat([self.weight.data, new_weight], dim=0)
    #     self.weight = nn.Parameter(updated_weight)
    #     self.new_weight = nn.Parameter(new_weight)
    #     self.out_features = 2 * self.out_features

class AdaptiveLayer2(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(AdaptiveLayer2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.step = 5
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features*self.step))
        self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(out_features, in_features)) for i in range(self.step)])
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.clf = nn.ParameterList([nn.Parameter(torch.Tensor(10, out_features)) for i in range(self.step)])
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and bias
        # nn.init.xavier_uniform_(self.weight)
        for i in range(self.step):
            nn.init.xavier_uniform_(self.weight[i])
            nn.init.xavier_uniform_(self.clf[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, step):

        ###################################### progressive
        # return torch.matmul(input, self.weight[:, :step*self.in_features].T) + self.bias
        weight = self.weight[0]
        if step > 1:
            for i in range(1, step):
                weight = torch.cat((weight, self.weight[i]), dim=1)

        middle = torch.matmul(input, weight.T.to(input.device)) + self.bias.to(input.device)
            
        return torch.matmul(middle, self.clf[step-1].T.to(input.device))
        #######################################

        # weight = self.weight[step-1]
        # return torch.matmul(input, weight.T.to(input.device)) + self.bias.to(input.device)
    
    def forward_test(self, input, step):
        weight = self.weight[0]
        if step > 1:
            for i in range(1, step):
                weight = weight + self.weight[i]
        return torch.matmul(input, weight.T.to(input.device)) + self.bias.to(input.device)


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        dims = [args.d_data] + args.GCN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GCN_args['dropout']
        self.mlp_layers = nn.ModuleList()
        self.step = 5
        # self.dim_2 = dims[1]*step
        
        # for l in range(len(dims) - 1):
        #     self.mlp_layers.append(nn.Linear(dims[l], dims[l + 1]))
        
        # for l in range(len(dims) - 1):
        #     self.mlp_layers.append(AdaptiveLayer(dims[l], dims[l + 1]))


        ####
        self.mlp_layers.append(AdaptiveLayer(dims[0], dims[1]))
        self.mlp_layers.append(AdaptiveLayer2(dims[1], dims[2]))

    
    def forward(self, g, features, step):
        """
        Forward pass for non-batched data.
        Ignores the graph structure (g) and only uses the features.
        """
        e_list = []  # To maintain compatibility with GCN output
        h = features
        for layer in self.mlp_layers[:-1]:
            h = layer(h, step)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.mlp_layers[-1](h, step)
        self.second_last_h = logits if len(self.mlp_layers) == 1 else h
        return logits, e_list

    def forward_batch(self, blocks, features, step):
        """
        Forward pass for batched data.
        Ignores the graph structure (blocks) and processes features.
        """
        e_list = []  # To maintain compatibility with GCN output
        h = features
        for layer in self.mlp_layers[:-1]:
            h = layer(h, step)
            # h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.mlp_layers[-1](h, step)
        # logits = self.mlp_layers[-1](h)
        self.second_last_h = logits if len(self.mlp_layers) == 1 else h
        return logits, e_list
    
    def forward_batch_test(self, blocks, features, step):
        e_list = []  # To maintain compatibility with GCN output
        h = features
        for layer in self.mlp_layers[:-1]:
            h = layer.forward_test(h, step)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.mlp_layers[-1].forward_test(h, step)
        self.second_last_h = logits if len(self.mlp_layers) == 1 else h
        return logits, e_list

    def reset_params(self):
        """
        Reset parameters of all MLP layers.
        """
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


class GIN(nn.Module):
    def __init__(self,
                 args,):
        super(GIN, self).__init__()
        dims = [args.d_data] + args.GIN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GIN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            lin = torch.nn.Linear(dims[l], dims[l+1])
            self.gat_layers.append(GINConv(lin, 'sum'))


    def forward(self, g, features):
        e_list = []
        h, e = self.gat_layers[0](g, features)
        x = F.relu(h)
        logits, e = self.gat_layers[1](g, x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h, e = self.gat_layers[0].forward_batch(blocks[0], features)
        x = F.relu(h)
        logits, e = self.gat_layers[1].forward_batch(blocks[1], x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GIN_original(nn.Module):
    def __init__(self, args, ):
        super().__init__()
        dims = [args.d_data] + args.GIN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GIN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims) - 1):
            lin = torch.nn.Linear(dims[l], dims[l + 1])
            self.gat_layers.append(GINConv(lin, 'sum'))

    def forward(self, g, features):
        e_list = []
        h, e = self.gat_layers[0](g, features)
        x = F.relu(h)
        # e_list = e_list + e
        logits, e = self.gat_layers[1](g, x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h, e = self.gat_layers[0].forward_batch(blocks[0], features)
        x = F.relu(h)
        logits, e = self.gat_layers[1].forward_batch(blocks[1], x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GCN(nn.Module):
    def __init__(self,
                 args):
        super(GCN, self).__init__()
        dims = [args.d_data] + args.GCN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GCN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            self.gat_layers.append(GCNLayer(dims[l], dims[l+1]))

    def forward(self, g, features):
        e_list = []
        h = features
        for layer in self.gat_layers[:-1]:
            h, e = layer(g, h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1](g, h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GAT(nn.Module):
    def __init__(self,
                 args,
                 heads,
                 activation):
        super(GAT, self).__init__()
        #self.g = g
        self.num_layers = args.GAT_args['num_layers']
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            args.d_data, args.GAT_args['num_hidden'], heads[0],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], False, None))
        # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[0]))
        self.norm_layers.append(PairNorm())
        
        # hidden layers
        for l in range(1, args.GAT_args['num_layers']):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                args.GAT_args['num_hidden'] * heads[l-1], args.GAT_args['num_hidden'], heads[l],
                args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], self.activation))
            # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[l]))
            self.norm_layers.append(PairNorm())
        # output projection

        self.gat_layers.append(GATConv(
            args.GAT_args['num_hidden'] * heads[-2], args.n_cls, heads[-1],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], None))

    def forward(self, g, inputs, save_logit_name = None):
        h = inputs
        e_list = []
        for l in range(self.num_layers):
            h, e = self.gat_layers[l](g, h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        # store for ergnn
        self.second_last_h = h
        # output projection
        logits, e = self.gat_layers[-1](g, h)
        #self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class Gate(nn.Module):
    def __init__(self, args):
        super(Gate, self).__init__()
        dims = [args.d_data] + args.GCN_args['h_dims'] + [5]
        self.dropout = args.GCN_args['dropout']
        self.gate_layers = nn.ModuleList()
        
        for l in range(len(dims) - 1):
            self.gate_layers.append(nn.Linear(dims[l], dims[l + 1]))

    def forward(self, g, features):
        """
        Forward pass for non-batched data.
        Ignores the graph structure (g) and only uses the features.
        """
        h = features
        for layer in self.gate_layers[:-1]:
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.gate_layers[-1](h)
        return logits

    def forward_batch(self, blocks, features):
        """
        Forward pass for batched data.
        Ignores the graph structure (blocks) and processes features.
        """
        h = features
        for layer in self.gate_layers[:-1]:
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.gate_layers[-1](h)
        return logits

    def reset_params(self):
        """
        Reset parameters of all MLP layers.
        """
        for layer in self.gate_layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


class GMM(nn.Module):
    def __init__(self, arg, n_components=14, pca_components=200, covariance_type='full', random_state=42):
        super(GMM, self).__init__()

        self.n_components = n_components
        self.pca_components = pca_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.pca = PCA(n_components=self.pca_components)
        self.gmm = GaussianMixture(n_components=self.n_components, 
                                   covariance_type=self.covariance_type, 
                                   random_state=self.random_state)
        self.reduced_data = None
        self.labels = None
    def forward_batch(self, blocks, features):
        self.reduced_data = self.pca.fit_transform(features.cpu().numpy())
        self.gmm.fit(self.reduced_data)

    def predict_cluster(self, features):
        reduced_new_data = self.pca.transform(features.cpu().numpy())
        cluster_labels = self.gmm.predict(reduced_new_data)

        return cluster_labels
    
    def get_cluster_means(self):
        return self.gmm.means_
    
    def pca_(self, features):
        return self.pca.transform(features.cpu().numpy())
    


class NormalizingFlowWithPCA(nn.Module):
    def __init__(self, args, pca_dim=200, num_flows=4, hidden_dim=256):
        """
        Normalizing Flow with PCA for high-dimensional data.

        Args:
            input_dim (int): Original dimensionality of input data (e.g., 8170).
            pca_dim (int): Reduced dimensionality after PCA (e.g., 500).
            num_flows (int): Number of flow layers.
            hidden_dim (int): Hidden layer size in coupling layers.
        """
        super(NormalizingFlowWithPCA, self).__init__()
        
        self.input_dim = args.d_data
        self.pca_dim = pca_dim
        self.num_flows = num_flows
        self.hidden_dim = hidden_dim
        
        # PCA for Dimensionality Reduction
        self.pca = PCA(n_components=pca_dim, svd_solver="full", whiten=True)  # Add `whiten=True`
        
        # Base distribution (Standard Normal in reduced space)
        base_distribution = StandardNormal(shape=[pca_dim])
        
        # Create a sequence of coupling and permutation layers
        transforms = []
        for _ in range(num_flows):
            # Add a permutation to shuffle features
            transforms.append(LULinear(features=pca_dim))
            
            # # Add an affine coupling layer with alternating binary mask
            # mask = create_alternating_binary_mask(features=pca_dim)
            # transform_net = lambda in_features, out_features: nets.ResidualNet(
            #     in_features=in_features,
            #     out_features=out_features,
            #     hidden_features=hidden_dim,
            #     num_blocks=2,
            #     activation=torch.nn.Softplus(),
            #     use_batch_norm=True
            # )
            # transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=transform_net))
            transforms.append(MaskedAffineAutoregressiveTransform(features=pca_dim, hidden_features=hidden_dim))

        # Define the full normalizing flow model
        self.flow = Flow(transform=CompositeTransform(transforms), distribution=base_distribution)

    def fit_pca(self, data):
        """
        Fit PCA on high-dimensional data.
        
        Args:
            data (torch.Tensor or numpy.ndarray): High-dimensional data (N, input_dim).
        """
        if isinstance(data, torch.Tensor):  # Check if it's a PyTorch tensor
            data = data.cpu().numpy()  # Move to CPU and convert to NumPy
        
        print(f"Fitting PCA on data of shape: {data.shape}")
        self.pca.fit(data)

    def transform_pca(self, data):
        """
        Transform high-dimensional data to lower dimensions using fitted PCA.

        Args:
            data (torch.Tensor or numpy.ndarray): High-dimensional data (N, input_dim).

        Returns:
            Transformed data (numpy.ndarray or torch.Tensor).
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()  # Move to CPU before PCA transformation
        
        return self.pca.transform(data)

    
    def inverse_pca(self, reduced_data):
        """
        Reconstruct data from PCA space to original space.

        Args:
            reduced_data (numpy.ndarray): Reduced dimension data (N, pca_dim).

        Returns:
            Reconstructed data (N, input_dim).
        """
        return self.pca.inverse_transform(reduced_data)
    
    def forward(self, x):
        """ Forward pass: Compute log probability after PCA transformation. """
        # if isinstance(x, torch.Tensor):
        #     x = x.cpu().numpy()  # Convert to numpy before PCA

        # x_pca = self.transform_pca(x)  # Ensure PCA transformation
        # x_pca = torch.tensor(x_pca, dtype=torch.float32).to(x.device)  # Convert back to tensor
        
        log_prob = self.flow.log_prob(x)
        log_prob = torch.clamp(log_prob, min=-1e4, max=1e4)

        return log_prob
    
    def sample(self, num_samples):
        """
        Generate new samples from the learned distribution.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            samples (torch.Tensor): Generated samples.
        """
        return self.flow.sample(num_samples)
    
    def density(self, x):
        """
        Compute the probability density of x under the model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            density (torch.Tensor): Probability density.
        """
        return torch.exp(self.flow.log_prob(x))
    
    def fit_flow(self, dataloader, epochs=10, lr=1e-3):
        """
        Train the Normalizing Flow model using maximum likelihood.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        opt = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                if isinstance(batch, list) or isinstance(batch, tuple):  # ✅ Ensure batch is a tensor
                    batch = batch[0]  # Extract the first element if it's a tuple/list
                batch = batch.to(torch.float32)  # Ensure correct dtype

                batch_pca = self.transform_pca(batch.cpu().numpy())  # Apply PCA
                batch_pca = torch.tensor(batch_pca, dtype=torch.float32).to(batch.device)  # Convert back to tensor

                opt.zero_grad()
                loss = -self.forward(batch_pca).mean()  # Negative log-likelihood
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}")
    

    def compute_density(self, features_):
        """
        Compute density values for all features using a trained Normalizing Flow model.

        Args:
            flow_model (NormalizingFlowWithPCA): Trained NF model.
            features (torch.Tensor or np.ndarray): Feature matrix (N, input_dim).

        Returns:
            density_values (np.ndarray): Computed density values (N,).
        """
        if isinstance(features_, torch.Tensor):
            features = features_.cpu().numpy()  # Convert to NumPy
        
        # Step 1: Transform features using PCA
        transformed_features = self.transform_pca(features)
        
        # Step 2: Convert to tensor for NF model
        transformed_features_tensor = torch.tensor(transformed_features, dtype=torch.float32).to(features_.device)

        # Step 3: Compute density values
        with torch.no_grad():
            density_values = self.density(transformed_features_tensor).cpu().numpy()
    
        return density_values
    
    def find_prototypes(self, features, labels, density_values):
        """
        Computes the prototype for each class using density values.
        
        :param features: Tensor of shape (N, D) representing node features (on CUDA).
        :param labels: Tensor of shape (N,) containing class labels (on CUDA).
        :param density_values: Tensor of shape (N,) with density scores from normalizing flow (on CUDA).
        
        :return: Dictionary mapping class labels to prototype vectors.
        """

        if isinstance(density_values, np.ndarray):
            density_values = torch.tensor(density_values, device=features.device, dtype=torch.float32)

        unique_classes = torch.unique(labels)
        prototypes = {}

        for cls in unique_classes:
            class_indices = (labels == cls).nonzero(as_tuple=True)[0]  # Get indices for class
            class_features = features[class_indices]
            print(class_features.tolist())
            class_densities = density_values[class_indices]

            if len(class_features) == 0:
                continue  # Skip empty classes

            # Ensure tensors are on the same device
            class_features = class_features.to('cuda')  # Move features to CUDA if needed
            class_densities = class_densities.to('cuda')  # Move densities to CUDA if needed
            class_densities = class_densities.cpu()  # Explicitly move to CPU

            # # Option 1: Density-Weighted Mean (Robust to Noise)
            # density_weights = class_densities / torch.sum(class_densities)  # Normalize
            # prototype = torch.sum(class_features * density_weights.unsqueeze(1), dim=0)

            # Option 2: Highest-Density Feature (Mode-Based)
            prototype = class_features[torch.argmax(class_densities)]

            prototypes[int(cls)] = prototype.detach().cpu()  # Move to CPU before storing

        return prototypes
