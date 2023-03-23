from cmath import pi
import pickle
import pandas as pd
from pathlib import Path
import rdkit.Chem as Chem
from pytorch_lightning.utilities.cloud_io import load as pl_load

import torch
from molbart.data.datasets import ReactionDataset

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

# checkpoint1 = pl_load("tb_logs/backward_prediction/version_143/checkpoints/last.ckpt", map_location=lambda storage, loc: storage)

# checkpoint2 = pl_load("tb_logs/backward_prediction/version_141/checkpoints/last.ckpt", map_location=lambda storage, loc: storage)

# print(checkpoint1["state_dict"])
# # checkpoint2 = torch.load('fuse_pretrain_mask_3layer.pt')
# print(checkpoint2["state_dict"])
# print(123)


# a = [[0] * 3] * 2
# b = [[1]] * 2
# print(b)
# a[:][0] = b
# print(a)


# @staticmethod
# def _pad_adj(adjs, pad_token):
#     pad_col_length = max([len(seq[0]) for seq in adjs])
#     pad_row_length = max([len(seq) for seq in adjs])
#     padded = []
#     for seq in adjs:
#         adj = []
#         for adj_node in seq:
#             adj.append(adj_node + ([pad_token] * (pad_col_length - len(adj_node))))
#         for i in range(pad_row_length - len(seq)):
#             adj.append([pad_token] * pad_col_length)
#         padded.append(adj)

#     return padded



import numpy as np
import matplotlib.pyplot as plt
 
#生成数据
x = np.linspace(0, np.pi, 10)
y = 5.0*np.cos(x)
 
#拟合曲线
fx = np.poly1d(np.polyfit(y,x,5))
dfx = fx.deriv()  # deriv()方法可得该格式函数的导函数
ddfx = dfx.deriv()
 
#求曲率半径
def calc_kappa(s):
    dx = dfx(s)
    ddx = ddfx(s)
    r = ((1 + (dx ** 2)) ** (3 / 2)) / ddx
    return r
 
snew = np.linspace(0, 10, 50)
k_buf = []
for i in snew:
    k = calc_kappa(i)
    k_buf.append(k)
k_buf = np.array(k_buf)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(snew[1:], k_buf[1:], s=1.0, color='red')
print(k_buf)
plt.show()