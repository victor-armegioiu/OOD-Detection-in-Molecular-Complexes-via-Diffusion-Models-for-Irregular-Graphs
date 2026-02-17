import pickle
import sys
from generate_ligands import sample_molecule_given_all_atom_graph
from moldiff.metrics import load_checkpoint
import pdb

def track_diffusion(graph_path: str = "../bioinformatics/data_extracted_graphs_atom_level/1a1e.pt"):
    model = load_checkpoint("/cluster/work/math/pbaertschi/molecular-diffusion/training_runs/RETRAIN_norm1_1128_084449_datasets/datasetmd_pdbbind_train/checkpoint_epoch_210.pt")
    samples = sample_molecule_given_all_atom_graph(
        graph_path, 
        model, 
        output_dir="test_output_multiframe",
        pocket_source="data_extracted_graphs",
        n_samples=5, 
        num_steps= 600,
        return_full_paths=True, 
        path_save_step = 10

    )
    # pdb.set_trace()



if __name__ == "__main__":
    import pickle

    track_diffusion()



    # with open("samples.pkl", "rb") as f:
    #     samples = pickle.load(f)

    # import sys
    # sys.exit(0)



