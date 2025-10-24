import argparse
from torch_geometric.utils import remove_self_loops
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.io as io
# from sklearn.neighbors import NearestNeighbors
import torch
from moldiff.constants import amino_acid_colors, atom_colors, atom_decoder, aa_decoder3

def padded_concat(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    """
    Concatenate two tensors vertically with appropriate zero padding.
    First tensor is padded at the end, second tensor is padded at the beginning.
    
    Args:
        arr1: First tensor to concatenate, will be padded at the end
        arr2: Second tensor to concatenate, will be padded at the beginning
    
    Returns:
        Concatenated tensor with shape (arr1.shape[0] + arr2.shape[0], arr1.shape[1] + arr2.shape[1])
    """
    n1, m1 = arr1.shape
    n2, m2 = arr2.shape
    
    # Create padded versions of both tensors
    arr1_padded = torch.nn.functional.pad(arr1, (0, m2), mode='constant', value=0)
    arr2_padded = torch.nn.functional.pad(arr2, (m1, 0), mode='constant', value=0)
    
    # Concatenate vertically
    return torch.cat([arr1_padded, arr2_padded], dim=0)


def visualize_graph(graph, 
                    title,
                    highlight=None,
                    highlight_symbol = 'x',
                    show_edges = False,
                    add_traces = [], 
                    linewidth=2, 
                    markersize=3, 
                    output_type="notebook"):


    if show_edges: 
        edge_list, _ = remove_self_loops(graph.edge_index)
        edge_list = edge_list.T.tolist()
    
        edges=[]
        for idx, pair in enumerate(edge_list):
            if (pair[1], pair[0]) not in edges: 
                edges.append((pair[0], pair[1]))


        # Prepare hoverinfo as a list of lists, round floats
        hoverinfo_edges = 'None'
        hoverinfo_nodes = graph.x[:,9:40].tolist()
        for l in range(len(hoverinfo_nodes)):
            hoverinfo_nodes[l] = [int(entry) if entry % 1 == 0 else round(entry,4) for entry in hoverinfo_nodes[l]]


    N = graph.x.shape[0]
    print(N)


    #MARKER COLORS AND LABELS
    #------------------------------------------------------------------------------------------------------------------------------------

    hoverinfo_nodes = [0 for i in range(N)]
    markercolor = [0 for i in range(N)]
    markersize = [markersize for i in range(N)]


    index, atomtypes = (graph.x[:,:10] == 1).nonzero(as_tuple=True) #identify the index of the first 1 in the feature matrix = atom type
    print(atomtypes.tolist())
    for idx, atomtype in zip(index.tolist(), atomtypes.tolist()):

        hoverinfo_nodes[idx] = atom_decoder[atomtype] # If the chemical element should be displayed
        #hoverinfo_nodes[idx] = int(markersize[idx]) # If the attention score should be displayed
        markercolor[idx] = atom_colors[atom_decoder[atomtype]]
        markersize[idx] = 20


    index, aa_types = (graph.x[:,10:] == 1).nonzero(as_tuple=True) #identify the index of the first 1 in the feature matrix = aa type
    print([aa for aa in aa_types.tolist()])
    aa_names = [aa_decoder3[i] for i in aa_types.tolist()]


    for idx, aa_type, aa_name in zip(index.tolist(), aa_types.tolist(), aa_names):

        hoverinfo_nodes[idx] = aa_decoder3[aa_type]# + f'({int(markersize[idx])})'
        markercolor[idx] = amino_acid_colors[aa_name]


    print(markercolor)
    print(markersize)
    #-----------------------------------------------------------------------------------------------------------------------------------


    #Marker shape based on atom type (Fe, Cl and the atoms in highligh as crosses, the rest as circles)
    not_ions = [0,1,2,3,4,5,6,7,8]
    symbols = ['x' if atom not in not_ions else 'circle' for atom in atomtypes]
    if highlight != None:
        symbols = [highlight_symbol if index in highlight else symbol for index, symbol in enumerate(symbols)]


    # Prepare the coordinates of the nodes and edges
    atomcoords = graph.pos.detach().numpy()

    Xn=[atomcoords[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[atomcoords[k][1] for k in range(N)]# y-coordinates
    Zn=[atomcoords[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]

    if show_edges:
        for e in edges:
            Xe+=[atomcoords[e[0]][0],atomcoords[e[1]][0], None]# x-coordinates of edge ends
            Ye+=[atomcoords[e[0]][1],atomcoords[e[1]][1], None]# y-coordinates of edge ends
            Ze+=[atomcoords[e[0]][2],atomcoords[e[1]][2], None]# z-coordinates of edge ends


        # Configure Plot, trace1 = edges, trace2 = nodes
        trace1=go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(color='rgb(50,50,50)', width=0.2),
                text=hoverinfo_edges,
                #textposition= 'middle center',
                hoverinfo = 'text'
                )

    trace2=go.Scatter3d(x=Xn,
                    y=Yn,
                    z=Zn,
                    mode='markers',
                    marker=dict(symbol=symbols,
                                size=markersize,
                                color=markercolor,
                                #colorscale = 'viridis',
                                line=dict(color='rgb(50,50,50)', width=0.2)
                                ),
                    text=hoverinfo_nodes,
                    hoverinfo='text'
                    )
    
    trace3=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='text',
                textfont=dict(size=16),
                text=hoverinfo_nodes,
                )

    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

    layout = go.Layout(
            title=title,
            width=2000,
            height=2000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
    )

    # Add the traces that are given in add_traces
    if show_edges:
        data = [trace1, trace2, trace3] + add_traces
    else: 
        data = [trace2, trace3] + add_traces


    fig = go.Figure(data=data, layout=layout)
    if output_type == "notebook":
        iplot(fig)
    elif output_type == "browser":
        fig.show(renderer="browser")
    else:
        raise ValueError("output_type must be either 'notebook' or 'browser'")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    sample = torch.load(args.path)

    # print(sample.ligand_coords.shape)
    # print(sample.pocket_coords.shape)
    # print(sample.ligand_features.shape)
    # print(sample.pocket_features.shape)
    print(sample.pocket_coords)

    sample.x = padded_concat(sample.ligand_features, sample.pocket_features)
    sample.pos = torch.cat([sample.ligand_coords, sample.pocket_coords], dim=0)
    print(sample.x.shape)
    print(sample.pos.shape)


    visualize_graph(sample, title=args.path, output_type="browser")