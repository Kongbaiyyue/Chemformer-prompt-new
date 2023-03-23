import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from functools import partial
from molbart.models.encoder import MyTransformerEncoder, MyPreNormEncoderLayer
from molbart.models.decoder import MyTransformerDecoder

from molbart.models.template_prompt import TPrompt

from molbart.models.util import (
    PreNormEncoderLayer,
    PreNormDecoderLayer,
    FuncLR
)


# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Abstract Models ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class _AbsTransformerModel(pl.LightningModule):
    def __init__(
        self,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule,
        warm_up_steps,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        self.pad_token_idx = pad_token_idx
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.num_steps = num_steps
        self.max_seq_len = max_seq_len
        self.schedule = schedule
        self.warm_up_steps = warm_up_steps
        self.dropout = dropout

        if self.schedule == "transformer":
            assert warm_up_steps is not None, "A value for warm_up_steps is required for transformer LR schedule"

        # Additional args passed in to **kwargs in init will also be saved
        self.save_hyperparameters()

        # These must be set by subclasses
        self.sampler = None
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())

    def forward(self, x):
        raise NotImplementedError()

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor)
        """

        raise NotImplementedError()

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        self.train()

        model_output = self.forward(batch)
        loss = self._calc_loss(batch, model_output)

        self.log("train_loss", loss, on_step=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]

        loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        mol_acc = torch.tensor(metrics["accuracy"], device=loss.device)
        invalid = torch.tensor(metrics["invalid"], device=loss.device)

        # Log for prog bar only
        self.log("mol_acc", mol_acc, prog_bar=True, logger=False, sync_dist=True)

        val_outputs = {
            "val_loss": loss,
            "val_token_acc": token_acc,
            "perplexity": perplexity,
            # "val_molecular_accuracy": mol_acc,
            # "val_invalid_smiles": invalid
        }
        return val_outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def test_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]

        loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.test_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        test_outputs = {
            "test_loss": loss.item(),
            "test_token_acc": token_acc,
            "test_perplexity": perplexity,
            "test_invalid_smiles": metrics["invalid"]
        }

        if self.test_sampling_alg == "greedy":
            test_outputs["test_molecular_accuracy"] = metrics["accuracy"]

        elif self.test_sampling_alg == "beam":
            test_outputs["test_molecular_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_1_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_2_accuracy"] = metrics["top_2_accuracy"]
            test_outputs["test_molecular_top_3_accuracy"] = metrics["top_3_accuracy"]
            test_outputs["test_molecular_top_5_accuracy"] = metrics["top_5_accuracy"]
            test_outputs["test_molecular_top_10_accuracy"] = metrics["top_10_accuracy"]

        else:
            raise ValueError(f"Unknown test sampling algorithm, {self.test_sampling_alg}")

        return test_outputs

    def test_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def configure_optimizers(self):
        params = self.parameters()
        optim = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))

        if self.schedule == "const":
            print("Using constant LR schedule.")
            const_sch = FuncLR(optim, lr_lambda=self._const_lr)
            sch = {"scheduler": const_sch, "interval": "step"}

        elif self.schedule == "cycle":
            print("Using cyclical LR schedule.")
            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}

        elif self.schedule == "transformer":
            print("Using original transformer schedule.")
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}

        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model ** -0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step ** -0.5, step * (self.warm_up_steps ** -1.5))
        return self.lr * mult * lr

    def _const_lr(self, step):
        if self.warm_up_steps is not None and step < self.warm_up_steps:
            return (self.lr / self.warm_up_steps) * step

        return self.lr

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz, device="cpu"):
        """ 
        Method copied from Pytorch nn.Transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode 
        """

        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]
        vocab_dist_output = model_output["token_output"]

        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)

        seq_lengths = inv_target_mask.sum(dim=0)
        exp = - (1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean()

    def _calc_token_acc(self, batch_input, model_output):
        token_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy

    def _avg_dicts(self, colls):
        complete_dict = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]

        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)


# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Pre-train Models --------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class BARTModel(_AbsTransformerModel):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(
            pad_token_idx,
            vocab_size, 
            d_model,
            num_layers, 
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs
        )
        
        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = MyPreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = MyTransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_norm = nn.LayerNorm(d_model)
        dec_layer = PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        # self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)
        self.decoder = MyTransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.loss_attn = nn.MSELoss(reduction='sum')
        self.loss_type_fn = nn.CrossEntropyLoss(reduction="none")
        self.log_softmax = nn.LogSoftmax(dim=2)

        # freeze encoder and decoder
        # self.emb.weight.requires_grad = False
        # for name, parameters in self.encoder.named_parameters():
        #     parameters.requires_grad = False
        # for name, parameters in self.decoder.named_parameters():
        #     parameters.requires_grad = False
        # self.token_fc.weight.requires_grad = False

        # prompt
        # self.conv_prefix_embeds= None
        # self.conv_prefix_proj = None
        # self.prompt_proj2 = None
        # self.conv_prefix_embeds = nn.Parameter(torch.empty(20, d_model))
        # nn.init.normal_(self.conv_prefix_embeds)
        # self.conv_prefix_proj = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, d_model)
        # )

        # self.prompt_proj2 = nn.Linear(d_model, num_layers * 3 * d_model)

        # num_hiddens = 256
        # self.prompt_model = TPrompt(d_model, num_hiddens, num_heads, num_layers, 3, vocab_size=vocab_size, n_prefix_conv=64)

        # self.prompt_model = None

        # self.reaction_type_model = ReactionTypeModel(
        #                     decode_sampler,
        #                     pad_token_idx,
        #                     vocab_size, 
        #                     d_model,
        #                     num_layers, 
        #                     num_heads,
        #                     d_feedforward,
        #                     lr,
        #                     weight_decay,
        #                     activation,
        #                     num_steps,
        #                     max_seq_len,
        #                     schedule="cycle",
        #                     warm_up_steps=None,
        #                     dropout=0.1,
        #                     **kwargs
        #                 )
        # self.type_token_fc = nn.Linear(d_model, 10)
    
        self.n_layer = num_layers
        self.n_head = num_heads
        self.head_dim = d_model//self.n_head
        self.d_model = d_model
        print("using prompt")
        self._init_params()

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """
        
        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)

        # with torch.no_grad():
        #     token_ids = x["type_tokens"]
        #     type_token = self.reaction_type_model(x)["type_smiles"]
        #     _, pred_ids = torch.max(type_token.float(), dim=1)
        #     token_acc_index = torch.eq(token_ids, pred_ids)
        #     # print(token_acc_index.sum().float()/pred_ids.shape[0])
        #     # print("pred_ids", pred_ids)
            
        #     # print("pred_ids:", pred_ids.shape)
        #     encoder_input[0, :] = (pred_ids + 262)

        batch_size = encoder_input.shape[1]
        
        # if self.emb.weight.requires_grad:
        #     # print("emb forward", self.emb.weight.requires_grad)
        #     self.emb.weight.requires_grad = False
        #     for name, parameters in self.encoder.named_parameters():
        #         parameters.requires_grad = False
        #     for name, parameters in self.decoder.named_parameters():
        #         parameters.requires_grad = False
        #     self.token_fc.weight.requires_grad = False
        
        # freeze gcn model
        # if self.prompt_model.graph_model.n_emb.weight.requires_grad == True:
        #     # print("gcn pram", self.prompt_model.graph_model.n_emb.weight.requires_grad)
        
        #     for name, parameters in self.prompt_model.graph_model.named_parameters():
        #         parameters.requires_grad = False
        
        encoder_embs = self._construct_input(encoder_input)
        decoder_embs = self._construct_input(decoder_input)
        
        # prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
        # prefix_embeds = prefix_embeds.expand(encoder_input.shape[1], -1, -1)
        # prompt_embeds = prefix_embeds
        # prompt_embeds = self.prompt_proj2(prompt_embeds)
        # # prompt_embeds = prompt_embeds.reshape(
        # #     encoder_input.shape[1], 20, self.n_layer, 3, self.n_head, self.head_dim
        # # ).permute(2, 3, 1, 0, 4, 5)  # (n_layer, n_block, prompt_len, batch_size, n_head, head_dim)
        # prompt_embeds = prompt_embeds.reshape(
        #     encoder_input.shape[1], 20, self.n_layer, 3, self.d_model
        # ).permute(2, 3, 1, 0, 4)  # (n_layer, n_block, prompt_len, batch_size, d_model)

        # graph prompt
        # atom = x["prods_atom"]
        # edge = x["prods_edge"]
        # length = x["lengths"]
        # adj = x["prods_adj"]
        # prompt_embeds = self.prompt_model(atom, edge, length, n_adj=adj)
        # prompt_embeds = self.prompt_model(batch_size=batch_size)
        # batch_size = prompt_embeds.shape[3]

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=encoder_embs.device)
        # memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask, prompt_embeds=prompt_embeds)
        
        memory, half_feature = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        model_output, att = self.decoder(
            decoder_embs,
            # memory[1:, :, :],
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_pad_mask,
            # memory_key_padding_mask=encoder_pad_mask.clone()[:, 1:]
            memory_key_padding_mask=encoder_pad_mask.clone()
        )
        token_output = self.token_fc(model_output)

        # add learning type
        # entity_len = atom.shape[1]
        # type_embeds = prompt_embeds.clone()[0, 0, :, :, :].permute(1, 0, 2)[:, -1, :].reshape(batch_size, -1)
        # type_smiles = self.prompt_model.token_fc(type_embeds)

        # predict reaction type
        # type_smiles = self.token_fc(memory[0, :, :])
        type_smiles = self.type_token_fc(half_feature[0, :, :])

        output = {
            "model_output": model_output,
            "token_output": token_output,
            "type_smiles": type_smiles.squeeze(),
            # "type_smiles": type_token,
            # "token_acc_index": token_acc_index
            "att": att
        }

        return output

    def encode(self, batch):
        """ Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)

        # type_token = self.reaction_type_model(batch)["type_smiles"]
        # _, pred_ids = torch.max(type_token.float(), dim=1)
        # # print("pred_ids", pred_ids)
        
        # # print("pred_ids:", pred_ids.shape)
        # encoder_input[0, :] = (pred_ids + 262)
        
        encoder_embs = self._construct_input(encoder_input)

        batch_size = encoder_input.shape[1]
        
        # prompt
        # prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
        # prefix_embeds = prefix_embeds.expand(encoder_input.shape[1], -1, -1)
        # prompt_embeds = prefix_embeds
        # prompt_embeds = self.prompt_proj2(prompt_embeds)
        # # prompt_embeds = prompt_embeds.reshape(
        # #     encoder_input.shape[1], 20, self.n_layer, 3, self.n_head, self.head_dim
        # # ).permute(2, 3, 1, 0, 4, 5)  # (n_layer, n_block, prompt_len, batch_size, n_head, head_dim)
        # prompt_embeds = prompt_embeds.reshape(
        #     encoder_input.shape[1], 20, self.n_layer, 3, self.d_model
        # ).permute(2, 3, 1, 0, 4)  # (n_layer, n_block, prompt_len, batch_size, d_model)
        
        # freeze gcn model
        # if self.prompt_model.gcn_model.n_emb.weight.requires_grad == True:
        #     print("gcn pram", self.prompt_model.gcn_model.n_emb.weight.requires_grad)
        
        #     for name, parameters in self.prompt_model.gcn_model.named_parameters():
        #         parameters.requires_grad = False

        # graph prompt
        # atom = batch["prods_atom"]
        # edge = batch["prods_edge"]
        # length = batch["lengths"]
        # adj = batch["prods_adj"]
        # prompt_embeds = self.prompt_model(atom, edge, length, n_adj=adj)
        # prompt_embeds = self.prompt_model(batch_size=batch_size)


        # model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask, prompt_embeds=prompt_embeds)

        model_output, half_feature = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        """ Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embs.device)

        model_output, att = self.decoder(
            decoder_embs, 
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask
        )
        token_output = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        type_tokens = batch_input["type_tokens"]
        # cross_mask = batch_input["cross_mask"]
        token_output = model_output["token_output"]
        # token_acc_index = model_output["token_acc_index"]
        type_smiles = model_output["type_smiles"]
        # att = model_output["att"]
        
        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)
        type_loss = self.loss_type_fn(type_smiles, type_tokens)
        seq_len, batch_size = tuple(tokens.size())
        # print(type_loss.shape)
        type_loss = type_loss.sum() / batch_size

        # compute cross attn loss
        # print("att shape", att.shape)
        # print("cross_mask shape", cross_mask.shape)
        # attns_shape = att.permute(0, 2, 1)[:, 1:-1, 1:]
        # attns_masked = attns_shape.masked_fill(~cross_mask.bool(),
        #                                         0.0)
        # cross_mask = cross_mask.float()
        # cross_loss = self.loss_attn(attns_masked, cross_mask)

        # print("type_loss", type_loss)
        loss = token_mask_loss + 0.1 * type_loss
        # loss = token_mask_loss
        # loss = type_loss
        # loss = cross_loss + token_mask_loss

        # return token_mask_loss
        return loss, token_mask_loss, 0.1 * type_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens
        # loss = loss.sum()

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_input = batch_input["encoder_input"]
        enc_mask = batch_input["encoder_pad_mask"]
        # prods_atom = batch_input["prods_atom"]
        # prods_adj = batch_input["prods_adj"]
        # prods_edge = batch_input["prods_edge"]
        # lengths = batch_input["lengths"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        encode_input = {
            "encoder_input": enc_input,
            "encoder_pad_mask": enc_mask,
            # "prods_atom": prods_atom,
            # "prods_adj": prods_adj,
            # "prods_edge": prods_edge,
            # "lengths": lengths
        }
        # memory = self.encode(encode_input)[1:, :, :]
        memory = self.encode(encode_input)
        # mem_mask = enc_mask.clone()[1:, :]
        mem_mask = enc_mask.clone()

        _, batch_size, _ = tuple(memory.size())

        decode_fn = partial(self._decode_fn, memory=memory, mem_pad_mask=mem_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, memory.device)

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, memory.device, k=self.num_beams)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask
        }
        model_output = self.decode(decode_input)
        return model_output

    def _calc_type_token_acc(self, batch_input, model_output):
        token_ids = batch_input["type_tokens"]
        token_output = model_output["type_smiles"]

        # target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=1)
        
        # print(token_ids.shape)
        # zero = torch.zeros_like(token_ids)
        
        # class_1 = torch.where(pred_ids == 264, pred_ids, zero)
        # class_2 = torch.where(pred_ids == 263, pred_ids, zero)
        # class_3 = torch.where(pred_ids == 265, pred_ids, zero)
        # class_4 = torch.where(pred_ids == 270, pred_ids, zero)
        # class_5 = torch.where(pred_ids == 268, pred_ids, zero)
        # class_6 = torch.where(pred_ids == 262, pred_ids, zero)
        # class_7 = torch.where(pred_ids == 266, pred_ids, zero)
        # class_8 = torch.where(pred_ids == 271, pred_ids, zero)
        # class_9 = torch.where(pred_ids == 267, pred_ids, zero)
        # class_10 = torch.where(pred_ids == 269, pred_ids, zero)

        # # correct_1 = torch.eq(token_ids, class_1).sum().float() / torch.eq(class_1, 264).sum().float()
        # # correct_2 = torch.eq(token_ids, class_2).sum().float() / torch.eq(class_1, 263).sum().float()
        # # correct_3 = torch.eq(token_ids, class_3).sum().float() / torch.eq(class_1, 265).sum().float()
        # # correct_4 = torch.eq(token_ids, class_4).sum().float() / torch.eq(class_1, 270).sum().float()
        # # correct_5 = torch.eq(token_ids, class_5).sum().float() / torch.eq(class_1, 268).sum().float()
        # # correct_6 = torch.eq(token_ids, class_6).sum().float() / torch.eq(class_1, 262).sum().float()
        # # correct_7 = torch.eq(token_ids, class_7).sum().float() / torch.eq(class_1, 266).sum().float()
        # # correct_8 = torch.eq(token_ids, class_8).sum().float() / torch.eq(class_1, 271).sum().float()
        # # correct_9 = torch.eq(token_ids, class_9).sum().float() / torch.eq(class_1, 267).sum().float()
        # # correct_10 = torch.eq(token_ids, class_10).sum().float() / torch.eq(class_1, 269).sum().float()
        # # print(correct_1)
        # correct_class = []
        # num = [264, 263, 265, 270, 268, 262, 266, 271, 267, 269]
        # token_class = [class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10]
        # for i, token_cls in enumerate(token_class):
        #     if torch.eq(token_cls, num[i]).sum() == 0:
        #         correct_class.append(0.)
        #     else:
        #         correct_class.append(torch.eq(token_ids, token_cls).sum().float() / torch.eq(token_cls, num[i]).sum().float())

        # correct_class = [correct_1, correct_2, correct_3, correct_4, correct_5, correct_6, correct_7, correct_8, correct_9, correct_10]
    
        correct_ids = torch.eq(token_ids, pred_ids)
        # correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()

        # total = target_mask.sum().float()
        total = token_ids.shape[0]

        accuracy = num_correct / total
        return accuracy

        # return accuracy, correct_class
    
    def test_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]

        loss, token_mask_loss, type_loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        reaction_type_acc = self._calc_type_token_acc(batch, model_output)
        # reaction_type_acc, correct_class = self._calc_type_token_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.test_sampling_alg)
        # print(mol_strs)
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        test_outputs = {
            "test_loss": loss.item(),
            "test_token_acc": token_acc,
            "reaction_type_acc": reaction_type_acc,
            "test_perplexity": perplexity,
            "test_invalid_smiles": metrics["invalid"]
        }
        # for i, class_t in enumerate(correct_class):
        #     test_outputs[str(i)] = class_t

        if self.test_sampling_alg == "greedy":
            test_outputs["test_molecular_accuracy"] = metrics["accuracy"]

        elif self.test_sampling_alg == "beam":
            test_outputs["test_molecular_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_1_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_2_accuracy"] = metrics["top_2_accuracy"]
            test_outputs["test_molecular_top_3_accuracy"] = metrics["top_3_accuracy"]
            test_outputs["test_molecular_top_5_accuracy"] = metrics["top_5_accuracy"]
            test_outputs["test_molecular_top_10_accuracy"] = metrics["top_10_accuracy"]

        else:
            raise ValueError(f"Unknown test sampling algorithm, {self.test_sampling_alg}")

        return test_outputs
    
    def validation_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]

        loss, token_mask_loss, type_loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        reaction_type_acc = self._calc_type_token_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        mol_acc = torch.tensor(metrics["accuracy"], device=loss.device)
        invalid = torch.tensor(metrics["invalid"], device=loss.device)

        # Log for prog bar only
        self.log("mol_acc", mol_acc, prog_bar=True, logger=False, sync_dist=True)
        self.log("reaction_type_acc", reaction_type_acc, prog_bar=True, logger=False, sync_dist=True)

        val_outputs = {
            "val_loss": loss,
            "val_token_mask_loss": token_mask_loss,
            "val_type_loss": type_loss,
            "val_token_acc": token_acc,
            "reaction_type_acc":reaction_type_acc,
            "perplexity": perplexity,
            "val_molecular_accuracy": mol_acc,
            "val_invalid_smiles": invalid
        }
        return val_outputs
    
    def validation_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        print("avg_outputs:", avg_outputs)
        path = "loss_logs/val_avg_type_loss_weight_0.1.txt"
        self.save_loss(avg_outputs, path)
        self._log_dict(avg_outputs)
    
    def training_step(self, batch, batch_idx):
        self.train()

        model_output = self.forward(batch)
        loss, token_mask_loss, type_loss = self._calc_loss(batch, model_output)
        # type_acc = self._calc_type_token_acc(batch, model_output)

        self.log("t_l", token_mask_loss, prog_bar=True, logger=False, sync_dist=True)
        self.log("ty_l", type_loss, prog_bar=True, logger=False, sync_dist=True)
        
        train_output = {
            "loss": loss,
            "train_loss": token_mask_loss,
            "type_loss": type_loss
            # "type_acc": type_acc
        }
        return train_output
    
    def training_epoch_end(self, outputs):
        # print(outputs[0].keys())
        avg_outputs = self._avg_dicts(outputs)
        path = "loss_logs/avg_type_loss_weight_0.1.txt"
        self.save_loss(avg_outputs, path)
        self._log_dict(avg_outputs)

    def save_loss(self, data, path):
        with open(path, "a") as f:
            f.write("epoch " + str(self.current_epoch) + ":\n")
            for key, val in data.items():
                if key == "train_loss" or key == "type_loss":
                    f.write(key + ": " + str(val) + "\n")
                elif key == "val_loss" or key == "val_token_mask_loss" or key == "val_type_loss":
                    f.write(key + ": " + str(val) + "\n")


class UnifiedModel(_AbsTransformerModel):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(
            pad_token_idx,
            vocab_size, 
            d_model,
            num_layers, 
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        enc_input = x["encoder_input"]
        enc_mask = x["encoder_pad_mask"]
        dec_input = x["decoder_input"]
        dec_mask = x["decoder_pad_mask"]
        att_mask = x["attention_mask"]

        model_input = torch.cat((enc_input, dec_input), dim=0)
        pad_mask = torch.cat((enc_mask, dec_mask), dim=0).transpose(0, 1)
        embs = self._construct_input(model_input)

        model_output = self.encoder(embs, mask=att_mask, src_key_padding_mask=pad_mask)
        token_output = self.token_fc(model_output)

        output = {
            "model_output": model_output,
            "token_output": token_output
        }

        return output

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        tgt_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, tgt_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size, _ = tuple(token_output.size())
        tgt_len, tgt_batch_size = tuple(target.size())

        assert seq_len == tgt_len
        assert batch_size == tgt_batch_size

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~target_mask
        num_tokens = inv_target_mask.sum()

        loss = loss * inv_target_mask
        loss = loss.sum() / num_tokens

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_token_ids = batch_input["encoder_input"]
        enc_pad_mask = batch_input["encoder_pad_mask"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        enc_seq_len, batch_size = tuple(enc_token_ids.size())
        self.sampler.max_seq_len = self.max_seq_len - enc_seq_len

        decode_fn = partial(self._decode_fn, enc_token_ids=enc_token_ids, enc_pad_mask=enc_pad_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, enc_token_ids.device)

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, enc_token_ids.device, k=self.num_beams)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, enc_token_ids, enc_pad_mask):
        # Strip off the start token for the decoded sequence
        dec_token_ids = token_ids[1:, :]

        enc_length, _ = tuple(enc_token_ids.shape)
        dec_length, _ = tuple(dec_token_ids.shape)
        att_mask = self._build_att_mask(enc_length - 1, dec_length + 1, device=dec_token_ids.device)

        model_input = {
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_pad_mask,
            "decoder_input": dec_token_ids,
            "decoder_pad_mask": pad_mask[1:, :],
            "attention_mask": att_mask
        }
        token_output = self.forward(model_input)["token_output"]
        token_probs = self.log_softmax(token_output)
        return token_probs

    def _build_att_mask(self, enc_length, dec_length, device="cpu"):
        seq_len = enc_length + dec_length
        enc_mask = torch.zeros((seq_len, enc_length), device=device)
        upper_dec_mask = torch.ones((enc_length, dec_length), device=device)
        lower_dec_mask = torch.ones((dec_length, dec_length), device=device).triu_(1)
        dec_mask = torch.cat((upper_dec_mask, lower_dec_mask), dim=0)
        mask = torch.cat((enc_mask, dec_mask), dim=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


class ReactionTypeModel(_AbsTransformerModel):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size, 
        d_model,
        num_layers, 
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(
            pad_token_idx,
            vocab_size, 
            d_model,
            num_layers, 
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs
        )
        
        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = MyPreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = MyTransformerEncoder(enc_layer, num_layers, norm=enc_norm)
        # enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        # self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_norm = nn.LayerNorm(d_model)
        dec_layer = PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.loss_type_fn = nn.CrossEntropyLoss(reduction="sum")
        self.log_softmax = nn.LogSoftmax(dim=2)

        # num_hiddens = 256
        # self.prompt_model = TPrompt(d_model, num_hiddens, num_heads, num_layers, 3, vocab_size=vocab_size, n_prefix_conv=64)
        self.type_token_fc = nn.Linear(d_model, 10)

        # self.prompt_model = None
        self.n_layer = num_layers
        self.n_head = num_heads
        self.head_dim = d_model//self.n_head
        self.d_model = d_model
        print("using prompt")
        self._init_params()

    def forward(self, x):
        """ Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model 
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """
        
        encoder_input = x["encoder_input"]
        # decoder_input = x["decoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)

        batch_size = encoder_input.shape[1]
        # decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)
        
        # if self.emb.weight.requires_grad:
        #     # print("emb forward", self.emb.weight.requires_grad)
        #     self.emb.weight.requires_grad = False
        #     for name, parameters in self.encoder.named_parameters():
        #         parameters.requires_grad = False
        #     for name, parameters in self.decoder.named_parameters():
        #         parameters.requires_grad = False
        #     self.token_fc.weight.requires_grad = False
        
        # freeze gcn model
        # if self.prompt_model.graph_model.n_emb.weight.requires_grad == True:
        #     # print("gcn pram", self.prompt_model.graph_model.n_emb.weight.requires_grad)
        
        #     for name, parameters in self.prompt_model.graph_model.named_parameters():
        #         parameters.requires_grad = False
        
        encoder_embs = self._construct_input(encoder_input)
        # decoder_embs = self._construct_input(decoder_input)

        # graph prompt
        # atom = x["prods_atom"]
        # edge = x["prods_edge"]
        # length = x["lengths"]
        # adj = x["prods_adj"]
        # prompt_embeds = self.prompt_model(atom, edge, length, n_adj=adj)
        # # prompt_embeds = self.prompt_model(None, None, None, batch_size=batch_size)
        # prompt_embeds = self.prompt_model(batch_size=batch_size)
        # batch_size = prompt_embeds.shape[3]

        # seq_len, _, _ = tuple(decoder_embs.size())
        # tgt_mask = self._generate_square_subsequent_mask(seq_len, device=encoder_embs.device)
        # memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask, prompt_embeds=prompt_embeds)
        
        # memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        model_output, half = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        # model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask, prompt_embeds=prompt_embeds)
        # model_output = self.decoder(
        #     decoder_embs,
        #     memory[1:, :, :],
        #     tgt_mask=tgt_mask,
        #     tgt_key_padding_mask=decoder_pad_mask,
        #     memory_key_padding_mask=encoder_pad_mask.clone()[:, 1:]
        # )
        # token_output = self.token_fc(model_output)

        # add learning type
        # entity_len = atom.shape[1]
        # type_embeds = prompt_embeds.clone()[0, 0, :, :, :].permute(1, 0, 2)[:, -1, :].reshape(batch_size, -1)
        # type_smiles = self.prompt_model.token_fc(type_embeds)

        type_smiles = self.type_token_fc(model_output[0, :, :])

        output = {
            "model_output": model_output,
            # "token_output": token_output,
            "type_smiles": type_smiles.squeeze()
        }

        return output

    def encode(self, batch):
        """ Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        
        encoder_embs = self._construct_input(encoder_input)

        # prompt
        # prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
        # prefix_embeds = prefix_embeds.expand(encoder_input.shape[1], -1, -1)
        # prompt_embeds = prefix_embeds
        # prompt_embeds = self.prompt_proj2(prompt_embeds)
        # # prompt_embeds = prompt_embeds.reshape(
        # #     encoder_input.shape[1], 20, self.n_layer, 3, self.n_head, self.head_dim
        # # ).permute(2, 3, 1, 0, 4, 5)  # (n_layer, n_block, prompt_len, batch_size, n_head, head_dim)
        # prompt_embeds = prompt_embeds.reshape(
        #     encoder_input.shape[1], 20, self.n_layer, 3, self.d_model
        # ).permute(2, 3, 1, 0, 4)  # (n_layer, n_block, prompt_len, batch_size, d_model)
        
        # freeze gcn model
        # if self.prompt_model.gcn_model.n_emb.weight.requires_grad == True:
        #     print("gcn pram", self.prompt_model.gcn_model.n_emb.weight.requires_grad)
        
        #     for name, parameters in self.prompt_model.gcn_model.named_parameters():
        #         parameters.requires_grad = False

        # graph prompt
        atom = batch["prods_atom"]
        edge = batch["prods_edge"]
        length = batch["lengths"]
        adj = batch["prods_adj"]
        prompt_embeds = self.prompt_model(atom, edge, length, n_adj=adj)


        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask, prompt_embeds=prompt_embeds)
        return model_output

    def decode(self, batch):
        """ Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embs.device)

        model_output = self.decoder(
            decoder_embs, 
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask
        )
        token_output = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        # tokens = batch_input["target"]
        # pad_mask = batch_input["target_mask"]
        type_tokens = batch_input["type_tokens"]
        # token_output = model_output["token_output"]
        type_smiles = model_output["type_smiles"]
        
        # token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)
        type_loss = self.loss_type_fn(type_smiles, type_tokens)
        # print("type_loss", type_loss)
        # loss = token_mask_loss + 1 * type_loss
        loss = type_loss

        # return token_mask_loss
        return loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        # loss = loss.sum() / num_tokens
        loss = loss.sum()

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """ Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """


        return 1

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask
        }
        model_output = self.decode(decode_input)
        return model_output

    def _calc_type_token_acc(self, batch_input, model_output):
        token_ids = batch_input["type_tokens"]
        token_output = model_output["type_smiles"]

        # target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=1)
        # print(pred_ids)
        
        # print(token_ids.shape)
        # zero = torch.zeros_like(token_ids) - 1
        
        # class_1 = torch.where(pred_ids == 0, pred_ids, zero)
        # class_2 = torch.where(pred_ids == 1, pred_ids, zero)
        # class_3 = torch.where(pred_ids == 2, pred_ids, zero)
        # class_4 = torch.where(pred_ids == 3, pred_ids, zero)
        # class_5 = torch.where(pred_ids == 4, pred_ids, zero)
        # class_6 = torch.where(pred_ids == 5, pred_ids, zero)
        # class_7 = torch.where(pred_ids == 6, pred_ids, zero)
        # class_8 = torch.where(pred_ids == 7, pred_ids, zero)
        # class_9 = torch.where(pred_ids == 8, pred_ids, zero)
        # class_10 = torch.where(pred_ids == 9, pred_ids, zero)
        

        # # correct_1 = torch.eq(token_ids, class_1).sum().float() / torch.eq(class_1, 264).sum().float()
        # # correct_2 = torch.eq(token_ids, class_2).sum().float() / torch.eq(class_1, 263).sum().float()
        # # correct_3 = torch.eq(token_ids, class_3).sum().float() / torch.eq(class_1, 265).sum().float()
        # # correct_4 = torch.eq(token_ids, class_4).sum().float() / torch.eq(class_1, 270).sum().float()
        # # correct_5 = torch.eq(token_ids, class_5).sum().float() / torch.eq(class_1, 268).sum().float()
        # # correct_6 = torch.eq(token_ids, class_6).sum().float() / torch.eq(class_1, 262).sum().float()
        # # correct_7 = torch.eq(token_ids, class_7).sum().float() / torch.eq(class_1, 266).sum().float()
        # # correct_8 = torch.eq(token_ids, class_8).sum().float() / torch.eq(class_1, 271).sum().float()
        # # correct_9 = torch.eq(token_ids, class_9).sum().float() / torch.eq(class_1, 267).sum().float()
        # # correct_10 = torch.eq(token_ids, class_10).sum().float() / torch.eq(class_1, 269).sum().float()
        # # print(correct_1)
        # correct_class = []
        # # num = [264, 263, 265, 270, 268, 262, 266, 271, 267, 269]
        # num_class = [1512, 1191, 564, 90, 65, 835, 459, 81, 184, 23]
        # token_class = [class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10]
        # num_count = 0
        # pred_count = 0
        # total_count = torch.tensor(0., device="cuda")
        # for i, token_cls in enumerate(token_class):
        #     if torch.eq(token_ids, i).sum() == 0:
        #         correct_class.append(0.)
        #     else:
        #         num_count += torch.eq(token_ids, i).sum()
        #         pred_count += torch.eq(token_ids, token_cls).sum()
        #         # correct_class.append(torch.eq(token_ids, token_cls).sum().float() / torch.eq(token_ids, i).sum())
        #         correct_class.append(torch.eq(token_ids, token_cls).sum().float() * 79)
        #         total_count += (torch.eq(token_ids, token_cls).sum().float() * 79)
        # correct_class.append(total_count)
        # for correct_cls in correct_class:
        #     if correct_cls < 1.0 - 0.0001 and correct_cls > 0. + 0.0001:
        #         print("error")
        # # correct_class = [correct_1, correct_2, correct_3, correct_4, correct_5, correct_6, correct_7, correct_8, correct_9, correct_10]
    
        correct_ids = torch.eq(token_ids, pred_ids)
        # correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()

        # total = target_mask.sum().float()
        total = token_ids.shape[0]

        accuracy = num_correct / total
        return accuracy

        # return accuracy, correct_class
    
    def test_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        # target_smiles = batch["target_smiles"]

        loss = self._calc_loss(batch, model_output)
        # token_acc = self._calc_token_acc(batch, model_output)
        reaction_type_acc = self._calc_type_token_acc(batch, model_output)
        # reaction_type_acc, correct_class = self._calc_type_token_acc(batch, model_output)
        # perplexity = self._calc_perplexity(batch, model_output)
        # mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.test_sampling_alg)
        # metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        test_outputs = {
            "test_loss": loss.item(),
            # "test_token_acc": token_acc,
            "reaction_type_acc": reaction_type_acc,
            # "test_perplexity": perplexity,
            # "test_invalid_smiles": metrics["invalid"]
        }
        # for i, class_t in enumerate(correct_class):
        #     test_outputs[str(i)] = class_t

        # if self.test_sampling_alg == "greedy":
        #     test_outputs["test_molecular_accuracy"] = metrics["accuracy"]

        # elif self.test_sampling_alg == "beam":
        #     test_outputs["test_molecular_accuracy"] = metrics["top_1_accuracy"]
        #     test_outputs["test_molecular_top_1_accuracy"] = metrics["top_1_accuracy"]
        #     test_outputs["test_molecular_top_2_accuracy"] = metrics["top_2_accuracy"]
        #     test_outputs["test_molecular_top_3_accuracy"] = metrics["top_3_accuracy"]
        #     test_outputs["test_molecular_top_5_accuracy"] = metrics["top_5_accuracy"]
        #     test_outputs["test_molecular_top_10_accuracy"] = metrics["top_10_accuracy"]

        # else:
        #     raise ValueError(f"Unknown test sampling algorithm, {self.test_sampling_alg}")

        return test_outputs
    
    def validation_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        # target_smiles = batch["target_smiles"]

        loss = self._calc_loss(batch, model_output)
        # token_acc = self._calc_token_acc(batch, model_output)
        reaction_type_acc = self._calc_type_token_acc(batch, model_output)
        # perplexity = self._calc_perplexity(batch, model_output)
        # mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
        # metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        # mol_acc = torch.tensor(metrics["accuracy"], device=loss.device)
        # invalid = torch.tensor(metrics["invalid"], device=loss.device)

        # Log for prog bar only
        # self.log("mol_acc", mol_acc, prog_bar=True, logger=False, sync_dist=True)
        self.log("reaction_type_acc", reaction_type_acc, prog_bar=True, logger=False, sync_dist=True)

        val_outputs = {
            "val_loss": loss,
            # "val_token_acc": token_acc,
            "reaction_type_acc":reaction_type_acc,
            # "perplexity": perplexity,
            # "val_invalid_smiles": invalid
        }
        return val_outputs