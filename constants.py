COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
}

atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'P': 7, 'I': 8, 'F': 9}
atom_decoder = ['C', 'N', 'O', 'S', 'B', 'Br', 'Cl', 'P', 'I', 'F']
aa_encoder1 = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
aa_decoder1 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_encoder3 = {'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9, 'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19, 'METAL': 20, 'UNK': 21}
aa_decoder3 = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'METAL']

metals = [
    # Alkali Metals
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    # Alkaline Earth Metals
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    # Transition Metals
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv',
    # Lanthanides
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    # Actinides
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 
    'Es', 'Fm', 'Md', 'No', 'Lr',
    # Post-Transition Metals
    'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Nh', 'Fl', 'Mc', 'Lv',
    # Half-Metals
    'As', 'Si', 'Sb', 'Te'
]

amino_acid_colors = {
    'ALA': 'rgb(0,0,0)', # Black
    'ARG': 'rgb(0,0,255)', # Blue
    'ASN': 'rgb(173,216,230)', # Light blue
    'ASP': 'rgb(255,0,0)', # Red
    'CYS': 'rgb(255,255,0)', # Yellow
    'GLN': 'rgb(144,238,144)', # Light green
    'GLU': 'rgb(139,0,0)', # Dark red
    'GLY': 'rgb(255,255,255)', # White
    'HIS': 'rgb(255,105,180)', # Pink
    'ILE': 'rgb(0,100,0)', # Dark green
    'LEU': 'rgb(0,128,0)', # Green
    'LYS': 'rgb(0,0,139)', # Dark blue
    'MET': 'rgb(255,165,0)', # Orange
    'PHE': 'rgb(105,105,105)', # Dark grey
    'PRO': 'rgb(216,191,216)', # Light purple
    'SER': 'rgb(255,182,193)', # Light pink
    'THR': 'rgb(128,0,128)', # purple
    'TRP': 'rgb(139,69,19)', # Brown
    'TYR': 'rgb(211,211,211)', # Light grey
    'VAL': 'rgb(204,204,0)', # Dark yellow
    'METAL': 'rgb(128,128,128)', # light grey
    'UNK': 'rgb(255,255,255)' # white
}

atom_colors = {'B':'rgb(65,105,225)', 
                'C':'rgb(34,139,34)', 
                'N':'rgb(0,0,255)', 
                'O':'rgb(255,0,0)', 
                'S':'rgb(255,120,0)', 
                'Br':'rgb(238,210,2)', 
                'Cl':'rgb(238,110,2)', 
                'P':'rgb(238,110,2)', 
                'F':'rgb(238,110,2)',
                'I':'rgb(238,110,2)'}