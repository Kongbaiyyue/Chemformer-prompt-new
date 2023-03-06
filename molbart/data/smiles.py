import argparse
import os
import pickle

import torch
from rdkit import Chem
from tqdm import tqdm

import numpy as np
import networkx as nx
import logging
from typing import List


BOND_TYPES = [None,
              Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]

BOND_STEREO = [Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREONONE]

def get_graph_from_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    rxn_graph = RxnGraph(prod_mol=mol)

    return rxn_graph


def get_bond_features(bond: Chem.Bond) -> List[int]:
    """Get bond features.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object
    """
    bt = bond.GetBondType()
    bond_features = [int(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bs = bond.GetStereo()
    bond_features.extend([int(bs == bond_stereo) for bond_stereo in BOND_STEREO])
    bond_features.extend([int(bond.GetIsConjugated()), int(bond.IsInRing())])

    return bond_features


def get_graph_features_from_smi(smi):

    atom_features = []
    bond_types = []
    edges = []

    global_node = [0] * 9
    atom_features.append(global_node)

    smi = smi.replace(' ', '')
    if not smi.strip():
        smi = "CC"          # hardcode to ignore
    graph = get_graph_from_smiles(smi).prod_mol

    mol = graph.mol
    assert mol.GetNumAtoms() == len(graph.G_dir)

    G = nx.convert_node_labels_to_integers(graph.G_dir, first_label=0)

    # node iteration to get sparse atom features
    for v, attr in G.nodes(data="label"):
        atom_feat = get_atom_features_sparse(mol.GetAtomWithIdx(v),
                                             use_rxn_class=False,
                                             rxn_class=graph.rxn_class)
        atom_features.append(atom_feat)
    bond_fea_num = 9
    nodes = len(G.nodes)
    # edges = torch.zeros(nodes+1, nodes+1, bond_fea_num)
    edges = [[[0] * bond_fea_num] * (nodes+1)] * (nodes+1)
    temp_adj = torch.ones(1, nodes)
    temp_adj2 = torch.ones(nodes+1, 1)
    adjacencys = torch.zeros(nodes, nodes)
    adjacencys = torch.cat((temp_adj, adjacencys), dim=0)
    adjacencys = torch.cat((temp_adj2, adjacencys), dim=1)

    # get bond type and edge
    for u, v, attr in G.edges(data='label'):
        bond_feat = torch.tensor(get_bond_features(mol.GetBondBetweenAtoms(u, v)))
        edges[u+1][v+1] = bond_feat
        edges[v+1][u+1] = bond_feat
        adjacencys[v+1][u+1] = 1
        adjacencys[u+1][v+1] = 1
        adjacencys[u+1][u+1] = 1
        adjacencys[v+1][v+1] = 1

    # atom_features = torch.tensor(atom_features, dtype=torch.float32)
    # bond_types = torch.tensor(bond_types, dtype=torch.int64)
    # edges = torch.tensor(edges, dtype=torch.int64)
    adjacencys = adjacencys.numpy().tolist()

    return atom_features, edges, adjacencys




# Symbols for different atoms
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '*', 'unk']
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LIST)}

MAX_NB = 10
DEGREES = list(range(MAX_NB))
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
HYBRIDIZATION_DICT = {hb: i for i, hb in enumerate(HYBRIDIZATION)}

FORMAL_CHARGE = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE)}

VALENCE = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE)}

NUM_Hs = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs)}

CHIRAL_TAG = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
              Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG)}

RS_TAG = ["R", "S", "None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG)}

BOND_TYPES = [None,
              Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_STEREO = [Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREONONE]

BOND_DELTAS = {-3: 0, -2: 1, -1.5: 2, -1: 3, -0.5: 4, 0: 5, 0.5: 6, 1: 7, 1.5: 8, 2: 9, 3: 10}
BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5]

RXN_CLASSES = list(range(10))

# ATOM_FDIM = len(ATOM_LIST) + len(DEGREES) + len(FORMAL_CHARGE) + len(HYBRIDIZATION) \
#             + len(VALENCE) + len(NUM_Hs) + 1
ATOM_FDIM = [len(ATOM_LIST), len(DEGREES), len(FORMAL_CHARGE), len(HYBRIDIZATION), len(VALENCE),
             len(NUM_Hs), len(CHIRAL_TAG), len(RS_TAG), 2]
# BOND_FDIM = 6
BOND_FDIM = 9
BINARY_FDIM = 5 + BOND_FDIM
INVALID_BOND = -1


def get_atom_features_sparse(atom: Chem.Atom, rxn_class: int = None, use_rxn_class: bool = False) -> List[int]:
    """Get atom features as sparse idx.

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit
    rxn_class: int, None
        Reaction class the molecule was part of
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    """
    feature_array = []
    symbol = atom.GetSymbol()
    symbol_id = ATOM_DICT.get(symbol, ATOM_DICT["unk"])
    feature_array.append(symbol_id)

    if symbol in ["*", "unk"]:
        padding = [999999999] * len(ATOM_FDIM) if use_rxn_class else [999999999] * (len(ATOM_FDIM) - 1)
        feature_array.extend(padding)

    else:
        degree_id = atom.GetDegree()
        if degree_id not in DEGREES:
            degree_id = 9
        formal_charge_id = FC_DICT.get(atom.GetFormalCharge(), 4)
        hybridization_id = HYBRIDIZATION_DICT.get(atom.GetHybridization(), 4)
        valence_id = VALENCE_DICT.get(atom.GetTotalValence(), 6)
        num_h_id = NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4)
        chiral_tag_id = CHIRAL_TAG_DICT.get(atom.GetChiralTag(), 2)

        rs_tag = atom.GetPropsAsDict().get("_CIPCode", "None")
        rs_tag_id = RS_TAG_DICT.get(rs_tag, 2)

        is_aromatic = int(atom.GetIsAromatic())
        feature_array.extend([degree_id, formal_charge_id, hybridization_id,
                              valence_id, num_h_id, chiral_tag_id, rs_tag_id, is_aromatic])

        if use_rxn_class:
            feature_array.append(rxn_class)

    return feature_array


def get_bond_features(bond: Chem.Bond) -> List[int]:
    """Get bond features.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object
    """
    bt = bond.GetBondType()
    bond_features = [int(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bs = bond.GetStereo()
    bond_features.extend([int(bs == bond_stereo) for bond_stereo in BOND_STEREO])
    bond_features.extend([int(bond.GetIsConjugated()), int(bond.IsInRing())])

    return bond_features



from rdkit import Chem
from typing import List, Tuple, Union


def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms:
                continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx():  # each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()


class RxnGraph:
    """
    RxnGraph is an abstract class for storing all elements of a reaction, like
    reactants, products and fragments. The edits associated with the reaction
    are also captured in edit labels. One can also use h_labels, which keep track
    of atoms with hydrogen changes. For reactions with multiple edits, a done
    label is also added to account for termination of edits.
    """

    def __init__(self,
                 prod_mol: Chem.Mol = None,
                 frag_mol: Chem.Mol = None,
                 reac_mol: Chem.Mol = None,
                 rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        prod_mol: Chem.Mol,
            Product molecule
        frag_mol: Chem.Mol, default None
            Fragment molecule(s)
        reac_mol: Chem.Mol, default None
            Reactant molecule(s)
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        if prod_mol is not None:
            self.prod_mol = RxnElement(mol=prod_mol, rxn_class=rxn_class)
        if frag_mol is not None:
            self.frag_mol = MultiElement(mol=frag_mol, rxn_class=rxn_class)
        if reac_mol is not None:
            self.reac_mol = MultiElement(mol=reac_mol, rxn_class=rxn_class)
        self.rxn_class = rxn_class

    def get_attributes(self, mol_attrs: Tuple = ('prod_mol', 'frag_mol', 'reac_mol')) -> Tuple:
        """
        Returns the different attributes associated with the reaction graph.

        Parameters
        ----------
        mol_attrs: Tuple,
            Molecule objects to return
        """
        return tuple(getattr(self, attr) for attr in mol_attrs if hasattr(self, attr))


class RxnElement:
    """
    RxnElement is an abstract class for dealing with single molecule. The graph
    and corresponding molecule attributes are built for the molecule. The constructor
    accepts only mol objects, sidestepping the use of SMILES string which may always
    not be achievable, especially for a unkekulizable molecule.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        self.mol = mol
        self.rxn_class = rxn_class
        self._build_mol()
        self._build_graph()

    def _build_mol(self) -> None:
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()
        self.num_bonds = self.mol.GetNumBonds()
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                            for atom in self.mol.GetAtoms()}
        self.idx_to_amap = {value: key for key, value in self.amap_to_idx.items()}

    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index(bond.GetBondType())
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        self.atom_scope = (0, self.num_atoms)
        self.bond_scope = (0, self.num_bonds)

    def update_atom_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the atom indices by the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        # Note that the self. reference to atom_scope is dropped to keep self.atom_scope non-dynamic
        if isinstance(self.atom_scope, list):
            atom_scope = [(st + offset, le) for st, le in self.atom_scope]
        else:
            st, le = self.atom_scope
            atom_scope = (st + offset, le)

        return atom_scope

    def update_bond_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the bond indices by the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        # Note that the self. reference to bond_scope is dropped to keep self.bond_scope non-dynamic
        if isinstance(self.bond_scope, list):
            bond_scope = [(st + offset, le) for st, le in self.bond_scope]
        else:
            st, le = self.bond_scope
            bond_scope = (st + offset, le)

        return bond_scope


class MultiElement(RxnElement):
    """
    MultiElement is an abstract class for dealing with multiple molecules. The graph
    is built with all molecules, but different molecules and their sizes are stored.
    The constructor accepts only mol objects, sidestepping the use of SMILES string
    which may always not be achievable, especially for an invalid intermediates.
    """

    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index(bond.GetBondType())
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        frag_indices = [c for c in nx.strongly_connected_components(self.G_dir)]
        self.mols = [get_sub_mol(self.mol, sub_atoms) for sub_atoms in frag_indices]

        atom_start = 0
        bond_start = 0
        self.atom_scope = []
        self.bond_scope = []

        for mol in self.mols:
            self.atom_scope.append((atom_start, mol.GetNumAtoms()))
            self.bond_scope.append((bond_start, mol.GetNumBonds()))
            atom_start += mol.GetNumAtoms()
            bond_start += mol.GetNumBonds()
