import argparse
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_SMILES import SmilesDataset, SmilesCollator
from model import BertModel, MaskLM


def train_step(x, y, char_weight):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get saved data/model path")
    parser.add_argument('--train_src', '-train_src', type=str, default='data/USPTO-50k_no_rxn/src-train.txt')
    parser.add_argument('--test_src', '-test_src', type=str, default='data/USPTO-50k_no_rxn/src-test.txt')
    parser.add_argument('--num_layers', '-num_layers', type=int, default=6)
    parser.add_argument('--d_model', '-d_model', type=int, default=256)
    parser.add_argument('--dff', '-dff', type=int, default=512)
    parser.add_argument('--num_heads', '-num_heads', type=int, default=8)
    parser.add_argument('--vocab_size', '-vocab_size', type=int, default=88)
    parser.add_argument('--max_length', '-max_length', type=int, default=256)
    parser.add_argument('--batch_size', '-batch_size', type=int, default=64)

    parser.add_argument('--adam_beta1', '-adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', '-adam_beta2', type=float, default=0.998)
    parser.add_argument('--lr', '-lr', type=float, default=0.001)
    parser.add_argument('--epochs', '-epochs', type=int, default=10)

    args = parser.parse_args()
    max_length = args.max_length
    batch_size = args.batch_size

    model = BertModel(num_layers=args.num_layers, d_model=args.d_model, dff=args.dff, num_heads=args.num_heads,
                      vocab_size=args.vocab_size)
    mask_model = MaskLM(args.vocab_size, args.dff, num_inputs=args.d_model, max_length=max_length)

    train_src = 'data/USPTO-50k_no_rxn/USPTO-50k_no_rxn.vocab.txt'
    train_dataset = SmilesDataset(args, args.train_src)
    test_dataset = SmilesDataset(args, args.test_src)

    train_data_collator = SmilesCollator(
        max_length=max_length
    )
    test_data_collator = SmilesCollator(
        max_length=max_length
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_data_collator,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    params2 = [p for p in mask_model.parameters() if p.requires_grad]
    para_num = 0
    for name, param in model.named_parameters():
        if name == 'encoder':
            para_num = param.nelement()
    print(para_num)
    params.extend(params2)
    betas = [args.adam_beta1, args.adam_beta2]
    optimizer = optim.Adam(params, lr=args.lr, betas=betas, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    epochs = args.epochs
    for epoch in range(epochs):
        start_time = time.time
        for step, batch in enumerate(train_dataloader):
            x = batch['x']
            y = batch['y']
            char_weight = batch['weight']

            optimizer.zero_grad()
            mask = torch.eq(x, 1).to(torch.float32)
            mask = mask.unsqueeze(1).unsqueeze(2)
            outputs = model(x, None, mask, training=True)
            mlm_Y_hat = mask_model(outputs, char_weight)

            weight = torch.nonzero(char_weight)
            y = y.reshape(-1)
            y = [y[idx[0]*max_length+idx[1]] for idx in weight]
            y = torch.tensor(y)
            loss = loss_fn(mlm_Y_hat, y)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(loss)
    print(loss)
    torch.save(model.state_dict(), 'Bert_smiles.pt')






