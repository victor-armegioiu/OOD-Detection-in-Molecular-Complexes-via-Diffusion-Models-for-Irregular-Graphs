# inspired by https://github.com/maabuu/posebusters/blob/main/posebusters/modules/intermolecular_distance.py#L16
# idea: for every ligand atom, get a gradient pulling them away from the neighbouring protein atoms

import torch

from .constants import atom_decoder
from torch_geometric.loader import DataLoader
from typing import List, Dict, Tuple, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from rdkit.Chem.rdchem import GetPeriodicTable


def fix_all_seeds(seed: int = 0):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



def create_batches_from_dataset(dataset_path: str, config: Dict) -> List[Dict]:
    """Create dataset from PDBbind dataset"""
    data = []
    dataset = torch.load(dataset_path)
    dataloader = DataLoader(dataset, 
                            batch_size=config['batch_size'], 
                            shuffle=True, 
                            follow_batch=['lig_coords', 'prot_coords']
                            )

    for batch in dataloader:
        # track COM for later addition in conditional sampling
        # com = batch.prot_coords.mean(dim=0, keepdim=True) # torch.cat([batch.lig_coords, batch.prot_coords], dim=0).mean(dim=0, keepdim=True)
        # assert len(com) == config["batch_size"]

        batch = {
            'ligand_coords': batch.lig_coords,
            'ligand_features': batch.lig_features,
            'ligand_mask': batch.lig_coords_batch,

            'pocket_coords': batch.prot_coords,
            'pocket_features': batch.prot_features,
            'pocket_mask': batch.prot_coords_batch,

            'sidechain_coords': batch.prot_sidechain_coords, 
            'sidechain_features': batch.prot_sidechain_features,

            'batch_size': config['batch_size'],
            'ids': batch.id
            # track COM for later addition in conditional sampling
            # 'pocket_com': com #torch.cat([batch.lig_coords, batch.prot_coords], dim=0).mean(dim=0, keepdim=True)
        }
        data.append(batch)


    return data



# clash_cutoff
class SidechainRepulsiveGuidance:

    def __init__(
            self,
            mode = "cutoff_relu", # neg_logsumexp, topk 
            search_distance: float = 4.0,
            radius_type: str = "vdw", 
            temperature: float = 0.5,
            clash_cutoff: float = 0.75, 
            k:int = 3
    ):
        # set distance params
        self.search_distance = search_distance
        self.radius_type = radius_type
        

        # set mode and needed scoring params
        if mode == "cutoff_relu":
            self.guidance_func = self._guideby_cutoff_relu
            self.temperature = temperature
            self.clash_cutoff = clash_cutoff
        elif mode == "neg_logsumexp":
            self.guidance_func = self._guideby_neg_logsumexp
            self.temperature = temperature
        elif mode == "topk":
            self.guidance_func = self._guideby_topk
            self.k = k
        else:
            raise ValueError(f"Unknown mode {mode}, must be one of cutoff_relu, neg_logsumexp or topk.")
        
        self.PERIODIC_TABLE = GetPeriodicTable()
    
    def __call__(
            
        self, 
        lig_coords: torch.Tensor,
        sidechain_coords: torch.Tensor,
        lig_features: torch.Tensor,
        sidechain_features: torch.Tensor
    ):
        
        
        with torch.enable_grad():
            lig_coords_grad = lig_coords.clone().detach().requires_grad_(True)

            relative_distance = self.proximal_relative_distance(
                lig_coords_grad, sidechain_coords, lig_features, sidechain_features
            )

            if not torch.isfinite(relative_distance).all():
                return torch.zeros_like(lig_coords_grad) # no guidance if there are no proximal sidechains

            score_per_lig_atom = self.guidance_func(relative_distance)
            score = score_per_lig_atom.sum()

            grad = torch.autograd.grad(
                score, lig_coords_grad, retain_graph=False, create_graph=False
            )[0]

        return grad.clone()
            


    def proximal_relative_distance(
        self, 
        lig_coords: torch.Tensor,
        sidechain_coords: torch.Tensor,
        lig_features: torch.Tensor,
        sidechain_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns a single scalar per step: mean (over ligand atoms) of a soft-min
        relative distance to proximal sidechain atoms.

        neg_logsumexp(d_ij) = -T * logsumexp(-d_ij / T)
        with d_ij = ||x_i - y_j|| / (r_i + r_j)
        """

        # radii (keep on same device)
        lig_types = torch.argmax(lig_features, dim=1).detach().cpu().tolist()
        sc_types = torch.argmax(sidechain_features, dim=1).detach().cpu().tolist()

        lig_radii = torch.tensor(
            [self._get_radius(atom_decoder[k]) for k in lig_types],
            device=lig_coords.device,
            dtype=lig_coords.dtype,
        )
        sc_radii = torch.tensor(
            [self._get_radius(atom_decoder[k]) for k in sc_types],
            device=lig_coords.device,
            dtype=lig_coords.dtype,
        )

        # pairwise distances + proximal sidechain mask
        dist = torch.cdist(lig_coords, sidechain_coords)  # (N_lig, N_sc)
        mask_sc = dist.min(dim=0).values <= self.search_distance
        if mask_sc.sum().item() == 0:
            return torch.ones(lig_coords.shape[0], sidechain_coords.shape[0]) * float("inf")
            # raise RuntimeError("No proximal sidechain atoms found — guidance undefined")

        dist = dist[:, mask_sc]
        sc_radii = sc_radii[mask_sc]

        rel = dist / (lig_radii[:, None] + sc_radii[None, :])  # (N_lig, N_sc_prox)

        return rel

    def _guideby_topk(
            self,
            relative_distance
        ):

        distance_score_per_atom = relative_distance.topk(
            k=self.k, dim=1, largest=False
            ).values.mean(dim=1)

        return distance_score_per_atom

    def _guideby_neg_logsumexp(
            self, 
            relative_distance
        ):


        distance_score_per_atom = - self.temperature * torch.logsumexp( # higher values become smaller relative to smaller values -> we want inverse of that!
            -relative_distance / self.temperature, dim=1    # low T leads to hard min, while greater T is avergage over many neighbours
        )
        return distance_score_per_atom

    def _guideby_cutoff_relu(
            self, 
           relative_distance: torch.Tensor
        ):

        gated_relative_distance_delta = torch.relu(self.clash_cutoff - relative_distance)

        distance_score_per_atom = self.temperature * torch.logsumexp( # higher values become smaller relative to smaller values -> we want inverse of that!
            -gated_relative_distance_delta / self.temperature, dim=1    # low T leads to hard min, while greater T is avergage over many neighbours
        )

        return distance_score_per_atom
    
    def _get_radius(self, a):
        if self.radius_type == "vdw":
            return self.PERIODIC_TABLE.GetRvdw(a)
        elif self.radius_type == "covalent":
            return self.PERIODIC_TABLE.GetRcovalent(a)
        else:
            raise ValueError(f"Unknown radius type {self.radius_type}. Valid values are 'vdw' and 'covalent'.")




# test anti-clash guidance
def test_anti_clash_guidance(batch, func, steps = 10, lig_offset = 3):

    # load a random graph
    device = "cpu"

    lig_coords = batch["ligand_coords"].to(device) + lig_offset
    lig_features = batch["ligand_features"].to(device)  # One-hot features
    pocket_coords = batch["pocket_coords"].to(device)
    pocket_features = batch["pocket_features"].to(device)  # One-hot features
    # new
    sidechain_coords =  batch["sidechain_coords"].to(device)
    sidechain_features = batch["sidechain_features"].to(device)

    # assert
    assert sidechain_coords.shape[1] == pocket_coords.shape[1] == lig_coords.shape[1] == 3
    assert sidechain_features.shape[1] == lig_features.shape[1] == 10
    assert pocket_features.shape[1] == 21

    ## clash guiding function
    radius_type = "vdw" # covalent
    search_distance = 4

    lig_coords_over_time = [lig_coords]

    for step in range(steps):
        lig_coords_step = lig_coords_over_time[-1].clone()
        update = func(
            lig_coords_step, 
            sidechain_coords,
            lig_features, 
            sidechain_features
        )
        lig_coords_over_time.append(lig_coords_step + update)

    return lig_coords_over_time





def plot_lig_coords_over_time_with_table(
    coords_list_dict,            # dict[str, list[Tensor(N,3)]]  (each list = steps)
    lig_features,                # Tensor(N_lig, onehot)
    guidance_class,              # guidance class to use for proximity map
    sidechain_coords=None,       # Tensor(N_sc,3) or None
    sidechain_features=None,     # Tensor(N_sc, onehot)
    out_file="traj_with_table.html",
    
):
    if sidechain_coords is None or sidechain_features is None:
        raise ValueError("sidechain_coords and sidechain_features are required for the distance table.")


    names = list(coords_list_dict.keys())
    if len(names) == 0:
        raise ValueError("coords_list_dict is empty.")

    # --- compute per-step metrics for each run ---
    metrics = {}  # name -> list[float]
    max_T = 0
    for name in names:
        traj = coords_list_dict[name]
        max_T = max(max_T, len(traj))
        vals = []
        with torch.no_grad():
            for step_idx, lig_coords_step in enumerate(traj):
                vals.append(
                    guidance_class.proximal_relative_distance(
                        lig_coords_step,
                        sidechain_coords,
                        lig_features,
                        sidechain_features
                    ).mean().item()
                )
            metrics[name] = vals

    # Normalize table length (pad with NaN if different run lengths)
    for name in names:
        if len(metrics[name]) < max_T:
            metrics[name] = metrics[name] + [float("nan")] * (max_T - len(metrics[name]))

    # compute improvement vs step0 (negative = improved, since distance lower is better)
    improvements = {}
    for name in names:
        base = metrics[name][0]
        improvements[name] = [(v - base) if (np.isfinite(v) and np.isfinite(base)) else float("nan") for v in metrics[name]]

    # --- build figure: 3D + table ---
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}, {"type": "table"}],
            [None,                             {"type": "xy"}, None],
        ],
        column_widths=[0.55, 0.25, 0.20],
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )


    # (A) 3D trajectories
    color_scales = [
        [[0.0, "rgb(255,200,150)"], [1.0, "rgb(180,0,0)"]],   # orange → red
        [[0.0, "rgb(180,210,255)"], [1.0, "rgb(0,0,160)"]],   # light → dark blue
        [[0.0, "rgb(180,255,180)"], [1.0, "rgb(0,120,0)"]],   # light → dark green
    ]

    for run_idx, name in enumerate(names):
        traj = coords_list_dict[name]
        arr = np.stack([x.detach().cpu().numpy() for x in traj])  # (T, N, 3)
        T, N, _ = arr.shape
        t_norm = np.linspace(0, 1, T)

        cs = color_scales[min(run_idx, 2)]  # cap at 3 schemes

        for atom_idx in range(N):
            xs, ys, zs = arr[:, atom_idx, 0], arr[:, atom_idx, 1], arr[:, atom_idx, 2]
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(width=4, color=t_norm, colorscale=cs),
                    opacity=0.9,
                    showlegend=(atom_idx == 0),
                    name=name,
                ),
                row=1, col=1
            )

    # sidechains (black, alpha=0.3)
    sc = sidechain_coords.detach().cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=sc[:, 0], y=sc[:, 1], z=sc[:, 2],
            mode="markers",
            marker=dict(size=3, color="black", opacity=0.3),
            name="Sidechain atoms",
            showlegend=True,
        ),
        row=1, col=1
    )

    fig.update_scenes(
        dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        row=1, col=1
    )

    steps = list(range(max_T))

    for run_idx, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=metrics[name],
                mode="lines+markers",
                name=f"{name} mean",
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_yaxes(
        title_text="Mean proximal relative distance",
        row=1, col=2,
    )

    for run_idx, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=improvements[name],
                mode="lines+markers",
                name=f"{name} Δ",
            ),
            row=2, col=2,
    )

    fig.update_xaxes(title_text="Step", row=2, col=2)
    fig.update_yaxes(
        title_text="Δ vs step₀ (↓ = improvement)",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="gray",
        row=2, col=2,
    )

    fig.update_layout(
    legend=dict(
        groupclick="toggleitem",
        tracegroupgap=8,
        )
    )




    # (B) table (step-wise mean proximal relative distance + Δ vs step0)
    step_col = [f"{i}" for i in range(max_T)]
    header = ["step"]
    cells = [step_col]

    for name in names:
        header.append(f"{name}: mean")
        cells.append([f"{v:.4f}" if np.isfinite(v) else "nan" for v in metrics[name]])
        header.append(f"{name}: Δvs0")
        cells.append([f"{dv:+.4f}" if np.isfinite(dv) else "nan" for dv in improvements[name]])

    fig.add_trace(
        go.Table(
            header=dict(values=header, align="left"),
            cells=dict(values=cells, align="left"),
        ),
        row=1, col=3
    )

    fig.update_layout(
        title=f"Trajectories + mean proximal relative distance per step",
        width=1200,
        height=800,
        legend=dict(itemsizing="constant"),
    )

    fig.write_html("tmp/" + out_file)
    print(f"[OK] Saved interactive plot to tmp/{out_file}")




if __name__ == "__main__":

    from moldiff.Dataset import PDBbind_Dataset

    fix_all_seeds(42)
    # unpack
    batch_size = 1 # TODO make sure it works across masks (scatter funtctionof gemometric?)
    graphs = create_batches_from_dataset("../bioinformatics/example_dataset_atom_level.pt", {"batch_size": batch_size})
    batch = next(iter(graphs))
    sidechain_coords =  batch["sidechain_coords"].to("cpu")
    lig_features = batch["ligand_features"].to("cpu")  # One-hot features
    # new
    sidechain_coords =  batch["sidechain_coords"].to("cpu")
    sidechain_features = batch["sidechain_features"].to("cpu")


    # instantiate functions:
    anti_clash_guidance_score_topk = SidechainRepulsiveGuidance(mode="topk")
    anti_clash_guidance_score_neg_logsumexp = SidechainRepulsiveGuidance(mode="neg_logsumexp")
    anti_clash_guidance_score_cutoff_relu = SidechainRepulsiveGuidance(mode="cutoff_relu")


    steps = 40

    plot_lig_coords_over_time_with_table(
        {"topk": test_anti_clash_guidance(batch, anti_clash_guidance_score_topk, steps=steps), 
         "neg_logsumexp": test_anti_clash_guidance(batch, anti_clash_guidance_score_neg_logsumexp, steps=steps), 
         "cutoff_relu": test_anti_clash_guidance(batch, anti_clash_guidance_score_cutoff_relu, steps=steps)
         }, 
         lig_features,
         anti_clash_guidance_score_cutoff_relu,
         sidechain_coords,
         sidechain_features

    )
    
    
    

