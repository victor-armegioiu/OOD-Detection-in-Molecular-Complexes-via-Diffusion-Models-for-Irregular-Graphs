import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.data import Data, Batch


class FeatureTransformMLP(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, dropout):
        super(FeatureTransformMLP, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, node_features):
        x = self.mlp(node_features)
        return self.dropout_layer(x)

class EdgeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals, dropout):
        super().__init__()
        self.residuals = residuals
        self.dropout_layer = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_node_f + n_edge_f, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.dropout_layer(out)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals, dropout):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        self.heads = 4

        self.conv = GATv2Conv(n_node_f, int(out_dim/self.heads), edge_dim=n_edge_f, heads=self.heads, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = F.relu(self.conv(x, edge_index, edge_attr))
        if self.residuals:
            out = out + x
        return out


class GlobalModel(torch.nn.Module):
    def __init__(self, n_node_f, glob_f_in, glob_f_hidden, glob_f_out, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.global_mlp = nn.Sequential(
            nn.Linear(n_node_f + glob_f_in, glob_f_hidden), 
            nn.ReLU(), 
            nn.Linear(glob_f_hidden, glob_f_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u, global_add_pool(x, batch=batch)], dim=1)
        out = self.dropout_layer(out)
        return self.global_mlp(out)



class GEMS18d(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GEMS18d, self).__init__()

        self.NodeTransform = FeatureTransformMLP(in_channels, 256, 64, dropout=dropout_prob)
        
        self.layer1 = self.build_layer( node_f=64, node_f_hidden=64, node_f_out=64, 
                                        edge_f=edge_dim, edge_f_hidden=64, edge_f_out=64,
                                        glob_f=384, glob_f_hidden=384, glob_f_out=384,
                                        residuals=False, dropout=conv_dropout_prob
                                        )
        
        self.node_bn1 = BatchNorm1d(64)
        self.edge_bn1 = BatchNorm1d(64)
        self.u_bn1 = BatchNorm1d(384)

        self.layer2 = self.build_layer( node_f=64, node_f_hidden=64, node_f_out=64,
                                        edge_f=64, edge_f_hidden=64, edge_f_out=64,
                                        glob_f=384, glob_f_hidden=384, glob_f_out=384,
                                        residuals=False, dropout=conv_dropout_prob
                                        )

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, 
                    node_f, node_f_hidden, node_f_out, 
                    edge_f, edge_f_hidden, edge_f_out,
                    glob_f, glob_f_hidden, glob_f_out,
                    residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, glob_f, glob_f_hidden, glob_f_out, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        x = self.NodeTransform(graphbatch.x)

        x, edge_attr, u = self.layer1(x, edge_index, graphbatch.edge_attr, u=graphbatch.lig_emb, batch=graphbatch.batch)
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)
        u = self.u_bn1(u)

        _, _, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        u = self.dropout_layer(u)

        # Fully-Connected Layers
        out = self.fc1(u)
        out = F.relu(out)
        out = self.fc2(out)
        return out


def main():
    
    num_graphs=4
    node_feature_dim=1148
    edge_feature_dim=20
    lig_emb_dim=384


    # Create random graph batch
    graphs = []
    for i in range(num_graphs):

        num_nodes_per_graph = random.randint(40, 100)
        num_edges_per_graph = random.randint(80, 200)

        x = torch.randn(num_nodes_per_graph, node_feature_dim)
        edge_index = torch.randint(0, num_nodes_per_graph, (2, num_edges_per_graph))
        edge_attr = torch.randn(num_edges_per_graph, edge_feature_dim)
        lig_emb = torch.randn(1, lig_emb_dim)

        # Print shapes of the tensors
        print(f"Graph {i+1}:")
        print(f"  Node features shape: {x.shape}")
        print(f"  Edge indices shape: {edge_index.shape}")
        print(f"  Edge attributes shape: {edge_attr.shape}")
        print(f"  Ligand embedding shape: {lig_emb.shape}")
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, lig_emb=lig_emb)
        graphs.append(graph)
    
    batch = Batch.from_data_list(graphs)
    print(f"\nData batch: {batch}")
    

    # Initialize model
    model = GEMS18d(
        dropout_prob=0.0,
        in_channels=node_feature_dim,
        edge_dim=edge_feature_dim,
        conv_dropout_prob=0.0
    )
    model.eval()

    # Forward pass
    with torch.no_grad():
        try:
            output = model(batch)
            print(f"\nForward pass successful!")
            print(f"Output shape: {output.shape}")
            print(f"Output values: {output.squeeze()}")
        except Exception as e:
            print(f"\nError during forward pass: {e}")
            raise


if __name__ == "__main__":
    main()