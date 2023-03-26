# from cmath import pi
# from email import header
import os
import pickle
import pandas as pd
from pathlib import Path
import rdkit.Chem as Chem
from pytorch_lightning.utilities.cloud_io import load as pl_load

# import torch
# from molbart.data.datasets import ReactionDataset

# from molbart.models.template_prompt import TPrompt

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


def plt_func():
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


def space_smi2mol(smi):
    chars = smi.strip().split(' ')
    smi = ''.join(chars)

    mol = Chem.MolFromSmiles(smi)
    return mol


# space_smi2mol("C C")
import sys
def get_uspto_full_pl():
    # path = Path(data_path)
    # df = pd.read_pickle(path)
    # print(df)
    # df = pd.DataFrame()
    df_dict = {"reactants_mol": [], "products_mol": [], "reaction_type": [], "set": []}
    root_path = "data/USPTO_full/"
    data_path = [("src-train.txt", "tgt-train.txt"), ("src-val.txt", "tgt-val.txt"), ("src-test.txt", "tgt-test.txt")]
    head_flag = True
    for path in data_path:
        data_cls = path[0].split('-')[1].split('.')[0]
        print(data_cls)
        print("memory size", sys.getsizeof(df_dict) / (1024 ** 2))
        with open(root_path + path[0], "r") as fs, open(root_path + path[1], "r") as ft:
            # pro_lines = fs.readlines()
            # rea_lines = ft.readlines()
            while True:
                pro_line = fs.readline()
                rea_line = ft.readline()
                if pro_line:
            # for i in range(len(pro_lines)):
                # df_dict["reactants_mol"].append(space_smi2mol(rea_lines[i]))
                # df_dict["products_mol"].append(space_smi2mol(pro_lines[i]))
                # df_dict["reactants_mol"].append(Chem.MolFromSmiles(''.join(rea_lines[i].strip().split(" "))))
                # df_dict["products_mol"].append(Chem.MolFromSmiles(''.join(pro_lines[i].strip().split(" "))))
                    # df_dict["reactants_mol"].append(Chem.MolFromSmiles(''.join(rea_line.strip().split(" "))))
                    # df_dict["products_mol"].append(Chem.MolFromSmiles(''.join(pro_line.strip().split(" "))))
                    df_dict["reactants_mol"].append(''.join(rea_line.strip().split(" ")))
                    df_dict["products_mol"].append(''.join(pro_line.strip().split(" ")))
                    df_dict["reaction_type"].append("<RX_10>")
                    df_dict["set"].append(data_cls)
                else:
                    break

                # if i % 10000 == 0:
                #     if head_flag:
                #         df = pd.DataFrame(df_dict)
                #         df.to_pickle("uspto_full.pickle", 'a')
                #         head_flag = False
                #     else:
                #         df = pd.DataFrame(df_dict, columns=None)
                #         df.to_pickle("uspto_full.pickle", 'a')
                #     with open('uspto_full.pickle', 'ab') as f:
                #         pickle.dumps(df)
                #     df_dict["reactants_mol"] = [] 
                #     df_dict["products_mol"] = []
                #     df_dict["reaction_type"] = []
                #     df_dict["set"] = []
    # print(df)
    df = pd.DataFrame(df_dict)
    # pickle.dumps(df)
    df.to_pickle("uspto_full.pickle")


def Three_token2smi():
    import json
    vocab_path = "bart_vocab_downstream.txt"
    sti = {}
    its = {}
    reaction_type = {"<RX_1>": 0, "<RX_2>": 1, "<RX_3>": 2, "<RX_4>": 3, "<RX_5>": 4, "<RX_6>": 5, "<RX_7>": 6, "<RX_8>": 7, "<RX_9>": 8, "<RX_10>": 9}
    t_its = {}
    for (k, v) in reaction_type.items():
        t_its[v] = k
    print(t_its)
    
    with open(vocab_path, "r") as f:
        vocab_strs = f.readlines()
        for i, token in enumerate(vocab_strs):
            token = token.strip()
            sti[token] = i
            its[i] = token
    root_path = "data/Pseudo_type_uspto_full/"
    data_path = "data/USPTO_full/Pseudo_type_uspto_full_only_reaction_type.txt"
    with open(data_path, "r") as f, open(root_path + 'src-train.txt', "a") as fs, open(root_path + 'tgt-train.txt', "a") as ft, open(root_path + 'type-train.txt', "a") as fy:
        lines = f.readlines()
        # prod, reac, type
        for i, tokens in enumerate(lines):
            if i % 3 == 0:
                tokens = tokens.split(":")[1].strip()
                tokens = json.loads(tokens)
                smi = ''
                for token in tokens:
                    if its[token] == '^':
                        continue
                    elif its[token] == '&' or its[token] == '<PAD>':
                        break
                    if smi == '':
                        smi += its[token]
                    else:
                        smi += (' ' + its[token])
                fs.write(smi + '\n')
                # print(smi)
                # break

            if i % 3 == 1:
                tokens = tokens.split(":")[1].strip()
                tokens = json.loads(tokens)
                smi = ''
                for token in tokens:
                    if its[token] == '^':
                        continue
                    elif its[token] == '&' or its[token] == '<PAD>':
                        break
                    if smi == '':
                        smi += its[token]
                    else:
                        smi += (' ' + its[token])
                ft.write(smi + '\n')
                # print(smi)
                # break

            if i % 3 == 2:
                tokens = tokens.split(":")[1].strip()
                
                type_i = int(tokens)
                type_s = t_its[type_i]
                fy.write(type_s + '\n')
                # print(type_s)
                # break

Three_token2smi()
# get_uspto_full_pl()

# path = "data/uspto_full.pickle"
# df = pd.read_pickle(path)
# print(df)

# a = [1, 2]
# print(str(a))