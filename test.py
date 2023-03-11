from cmath import pi
import pickle
import pandas as pd
from pathlib import Path
import rdkit.Chem as Chem
from pytorch_lightning.utilities.cloud_io import load as pl_load

import torch

from molbart.models.template_prompt import TPrompt

data_path = "data/uspto_50.pickle"

def combine_smiles():
    path = Path(data_path)
    df = pd.read_pickle(path)
    # pd.to_pickle()
    reactants = df["reactants_mol"].tolist()
    products = df["products_mol"].tolist()
    type_tokens = df["reaction_type"].tolist()
    set_type = df["set"].tolist()
    print(reactants[0])
    smiles = []
    for i, smile in enumerate(reactants):
        smiles.append([Chem.MolToSmiles(reactants[i]), set_type[i]])
    for i, smile in enumerate(products):
        smiles.append([Chem.MolToSmiles(products[i]), set_type[i]])
    # (smiles.append([smile]) for smile in products)
    print(len(smiles))
    print(smiles[0])
    print(smiles[1])
    df = pd.DataFrame(data=smiles, columns=["smiles", "set"])
    # print(df)
    df.to_pickle("data/uspto_50_pretrain")
    # print(len(reactants))

# combine_smiles()
# data_path = "data/uspto_50_pretrain"
# path = Path(data_path)
# df = pd.read_pickle(path)
# mol = df["reactants_mol"].tolist()[0]
# print(type(mol))
# print(mol)
# print(Chem.MolToSmiles(mol))

# reactants = df["reactants_mol"].tolist()
# set_type = df["set"].tolist()
# smiles = []
# for i, smile in enumerate(reactants):
#     smiles.append([reactants[i], set_type[i]])
# print(smiles)

# a = torch.ones((1, 5))
# b = torch.zeros((3, 5))
# c = torch.cat([a, b], dim=-2)
# print(c.shape)

checkpoint1 = pl_load("tb_logs/backward_prediction/version_80_freeze_fuse_graph_freeze_che_300epoch_120lenPrompt/checkpoints/last.ckpt", map_location=lambda storage, loc: storage)

# checkpoint2 = pl_load("models/combined/step=1000000.ckpt", map_location=lambda storage, loc: storage)

print(checkpoint1["state_dict"])
checkpoint2 = torch.load('fuse_pretrain_mask_3layer.pt')
# print(checkpoint2["state_dict"])
print(123)

model = TPrompt(512, 256, 8 , 6, 3, n_prefix_conv=256)
model.load_state_dict(torch.load('fuse_pretrain_mask_3layer.pt'), strict=False)
print(321)