import numpy as np
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, AllChem
from SA_Score.sascorer import calculateScore
import matplotlib.pyplot as plt
from moldiff.metrics import build_molecule
from copy import deepcopy

# Try to import seaborn, but don't fail if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class CategoricalDistribution:
    EPS = 1e-10

    def __init__(self, histogram_dict, mapping):
        histogram = np.zeros(len(mapping))
        for k, v in histogram_dict.items():
            histogram[mapping[k]] = v

        # Normalize histogram
        self.p = histogram / histogram.sum()
        self.mapping = deepcopy(mapping)

    def kl_divergence(self, other_sample):
        sample_histogram = np.zeros(len(self.mapping))
        for x in other_sample:
            # sample_histogram[self.mapping[x]] += 1
            sample_histogram[x] += 1

        # Normalize
        q = sample_histogram / sample_histogram.sum()

        return -np.sum(self.p * np.log(q / self.p + self.EPS))
    

def rdmol_to_smiles(rdmol):
    mol = Chem.Mol(rdmol)
    Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None,
                 connectivity_thresh=1.0):
        self.atom_decoder = dataset_info['atom_decoder']
        if dataset_smiles_list is not None:
            dataset_smiles_list = set(dataset_smiles_list)
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info
        self.connectivity_thresh = connectivity_thresh

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        if len(generated) < 1:
            return [], 0.0

        valid = []
        for mol in generated:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                continue

            valid.append(mol)

        return valid, len(valid) / len(generated)

    def compute_connectivity(self, valid):
        """ Consider molecule connected if its largest fragment contains at
        least x% of all atoms, where x is determined by
        self.connectivity_thresh (defaults to 100%). """
        if len(valid) < 1:
            return [], 0.0

        connected = []
        connected_smiles = []
        for mol in valid:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = \
                max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if largest_mol.GetNumAtoms() / mol.GetNumAtoms() >= self.connectivity_thresh:
                smiles = rdmol_to_smiles(largest_mol)
                if smiles is not None:
                    connected_smiles.append(smiles)
                    connected.append(largest_mol)

        return connected, len(connected_smiles) / len(valid), connected_smiles

    def compute_uniqueness(self, connected):
        """ valid: list of SMILES strings."""
        if len(connected) < 1 or self.dataset_smiles_list is None:
            return [], 0.0

        return list(set(connected)), len(set(connected)) / len(connected)

    def compute_novelty(self, unique):
        if len(unique) < 1:
            return [], 0.0

        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate_rdmols(self, rdmols):
        valid, validity = self.compute_validity(rdmols)
        print(f"Validity over {len(rdmols)} molecules: {validity * 100 :.2f}%")

        connected, connectivity, connected_smiles = \
            self.compute_connectivity(valid)
        print(f"Connectivity over {len(valid)} valid molecules: "
              f"{connectivity * 100 :.2f}%")

        unique, uniqueness = self.compute_uniqueness(connected_smiles)
        print(f"Uniqueness over {len(connected)} connected molecules: "
              f"{uniqueness * 100 :.2f}%")

        _, novelty = self.compute_novelty(unique)
        print(f"Novelty over {len(unique)} unique connected molecules: "
              f"{novelty * 100 :.2f}%")

        return [validity, connectivity, uniqueness, novelty], [valid, connected]

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """

        rdmols = [build_molecule(*graph, self.dataset_info)
                  for graph in generated]
        return self.evaluate_rdmols(rdmols)


class MoleculeProperties:
    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        return round((10 - sa) / 9, 2)  # from pocket2mol

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_lipinski(rdmol):
        # Compute Lipinski's rule-of-five count (number of satisfied rules)
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        logp = Crippen.MolLogP(rdmol)
        rule_4 = (logp >= -2) and (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        # Return as plain Python int (0..5)
        return int(rule_1) + int(rule_2) + int(rule_3) + int(rule_4) + int(rule_5)

    @staticmethod
    def calculate_basic_properties(rdmol):
        """Calculate basic molecular properties."""
        return {
            'num_atoms': rdmol.GetNumAtoms(),
            'num_bonds': rdmol.GetNumBonds(),
            'num_rings': rdmol.GetRingInfo().NumRings(),
            'ring_sizes': [len(ring) for ring in rdmol.GetRingInfo().AtomRings()],
            'num_rotatable_bonds': Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol),
            'num_fragments': len(Chem.rdmolops.GetMolFrags(rdmol)),
            'molecular_weight': Descriptors.ExactMolWt(rdmol),
            'polar_surface_area': Descriptors.TPSA(rdmol),
            'num_aromatic_rings': Chem.rdMolDescriptors.CalcNumAromaticRings(rdmol),
            'num_hbd': Lipinski.NumHDonors(rdmol),
            'num_hba': Lipinski.NumHAcceptors(rdmol)
        }

    @classmethod
    def calculate_diversity(cls, pocket_mols):
        if len(pocket_mols) < 2:
            return 0.0

        div = 0
        total = 0
        for i in range(len(pocket_mols)):
            for j in range(i + 1, len(pocket_mols)):
                div += 1 - cls.similarity(pocket_mols[i], pocket_mols[j])
                total += 1
        return div / total

    @staticmethod
    def similarity(mol_a, mol_b):
        fp1 = Chem.RDKFingerprint(mol_a)
        fp2 = Chem.RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    @staticmethod
    def get_list_from_sdf(sdf_file):
        """Read molecules from an SDF file."""
        suppl = Chem.SDMolSupplier(sdf_file)
        return [mol for mol in suppl if mol is not None]

    def evaluate_mol(self, rdmol):
        """Evaluate a single molecule and return all properties as a dictionary."""
        try:
            Chem.SanitizeMol(rdmol)
        except:
            return None

        properties = self.calculate_basic_properties(rdmol)
        properties.update({
            'qed': self.calculate_qed(rdmol),
            'sa': self.calculate_sa(rdmol),
            'logp': self.calculate_logp(rdmol),
            'lipinski': self.calculate_lipinski(rdmol)
        })
        return properties

    def evaluate(self, pocket_rdmols):
        """
        Run full evaluation on a list of lists of RDKit molecules
        Returns a dictionary of property lists and mean values
        """
        all_properties = []
        for pocket in tqdm(pocket_rdmols):
            pocket_properties = []
            for mol in pocket:
                props = self.evaluate_mol(mol)
                if props is not None:
                    pocket_properties.append(props)
            if pocket_properties:
                all_properties.append(pocket_properties)

        # Calculate diversity per pocket
        per_pocket_diversity = [self.calculate_diversity(pocket) for pocket in pocket_rdmols]

        # Flatten and calculate statistics (coerce numeric properties to float arrays)
        flattened_properties = {}
        if len(all_properties) > 0 and len(all_properties[0]) > 0:
            for prop in all_properties[0][0].keys():
                values = [p.get(prop) for pocket in all_properties for p in pocket]
                # Try to coerce to numeric float array; skip if impossible (e.g., lists)
                try:
                    arr = np.array(values, dtype=float)
                except Exception:
                    # non-numeric property (e.g., ring_sizes -> lists); skip
                    continue
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                print(f"{prop}: {mean:.3f} ± {std:.2f}")
                flattened_properties[prop] = {
                    'values': arr.tolist(),
                    'mean': mean,
                    'std': std
                }

        print(f"Diversity: {np.mean(per_pocket_diversity):.3f} ± {np.std(per_pocket_diversity):.2f}")
        
        return {
            'per_molecule': all_properties,
            'flattened': flattened_properties,
            'diversity': per_pocket_diversity
        }

    def plot_property_distributions(self, generated_properties_list, reference_mol=None, labels=None, ref_label="Reference"):
        """
        Plot histograms of molecular properties for one or multiple generated-property datasets.

        Args:
            generated_properties_list: either a single evaluation dict (from evaluate()) or
                a list of such dicts. Each dict must contain a 'flattened' key with numeric properties.
            reference_mol: Optional RDKit molecule for comparison (single molecule)
            labels: Optional list of labels for the generated property sets (same length as generated_properties_list)
            ref_label: Label used for the reference molecule line
        """
        # Normalize input to a list of dicts
        if isinstance(generated_properties_list, dict):
            gen_list = [generated_properties_list]
        else:
            gen_list = list(generated_properties_list)

        if labels is None:
            labels = [f"Set {i+1}" for i in range(len(gen_list))]

        # Evaluate reference properties once
        ref_props = self.evaluate_mol(reference_mol) if reference_mol is not None else None

        # Determine common numeric properties to plot (intersection across datasets)
        base_keys = set(gen_list[0].get('flattened', {}).keys())
        for gp in gen_list[1:]:
            base_keys &= set(gp.get('flattened', {}).keys())
        props_to_plot = sorted([p for p in base_keys])

        if not props_to_plot:
            raise ValueError("No numeric properties found to plot in generated properties.")

        n_cols = 5
        n_rows = (len(props_to_plot) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Choose colors
        if HAS_SEABORN:
            colors = sns.color_palette(n_colors=len(gen_list))
        else:
            cmap = plt.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(gen_list))]

        for idx, prop in enumerate(props_to_plot):
            ax = axes[idx]
            for ds_idx, gp in enumerate(gen_list):
                values = gp['flattened'][prop]['values']
                # plot as histogram overlay; use alpha for visibility
                if HAS_SEABORN:
                    sns.histplot(values, ax=ax, label=labels[ds_idx], color=colors[ds_idx], stat='count', element='step', fill=True, alpha=0.4)
                else:
                    ax.hist(values, bins=len(props_to_plot)//2, alpha=0.45, label=labels[ds_idx], color=colors[ds_idx])

            if ref_props is not None and prop in ref_props:
                ax.axvline(ref_props[prop], color='k', linestyle='--', label=ref_label)

            ax.set_title(prop)
            ax.set_xlabel(prop)
            ax.set_ylabel('Count')
            ax.legend()

        # Remove empty subplots
        for idx in range(len(props_to_plot), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        return fig
    
if __name__ == "__main__":
    # Initialize the class
    mp = MoleculeProperties()

    # Read molecules from an SDF file
    sampled_mols = mp.get_list_from_sdf("1a1e_our_model.sdf")
    sampled_mols_diffsbdd = mp.get_list_from_sdf("1a1e_mol.sdf")
    reference_mol = mp.get_list_from_sdf("example_dataset/1a1e.sdf")[0]  # Get first molecule

    # Evaluate properties
    # Note: wrap single molecule list in another list to match expected format
    print("=========Evaluate Ours ==========")
    properties = mp.evaluate([sampled_mols])
    print("========= Evaluate DiffSBDD =========")
    properties_diffsbdd = mp.evaluate([sampled_mols_diffsbdd])

    # Plot distributions with reference
    fig = mp.plot_property_distributions([properties, properties_diffsbdd], reference_mol=reference_mol, labels = ["Ours", "DiffSBDD"])
    # Save the figure to the root directory
    plt.savefig('molecular_property_distributions.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

    # Access specific properties
    mean_mw = properties['flattened']['molecular_weight']['mean']
    all_logp_values = properties['flattened']['logp']['values']
    per_mol_properties = properties['per_molecule'][0]  # List of dictionaries, one per molecule