import os
import argparse

import molbart.util as util
from molbart.decoder import DecodeSampler
from molbart.models.pre_train import BARTModel, ReactionTypeModel, UnifiedModel

import torch.nn as nn
import torch
from pytorch_lightning.utilities.cloud_io import load as pl_load
from molbart.models.template_prompt import TPrompt


# Default training hyperparameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_ACC_BATCHES = 8
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EPOCHS = 50
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SCHEDULE = "cycle"
DEFAULT_AUGMENT = "None"
DEFAULT_AUG_PROB = 0.5
DEFAULT_WARM_UP_STEPS = 8000
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = 24
DEFAULT_LIMIT_VAL_BATCHES = 1.0
DEFAULT_DOWNSTREAM_VOCAB_PATH = "bart_vocab_downstream.txt"
DEFAULT_MODEL_TYPE = "bart"


def load_rand_model(args, extra_args, sampler, vocab_size, total_steps, pad_token_idx):
    if args.model_type == "bart":
        model = BARTModel(
            sampler,
            pad_token_idx,
            vocab_size,
            args.d_model,
            args.num_layers,
            args.num_heads,
            args.d_feedforward,
            args.lr,
            DEFAULT_WEIGHT_DECAY,
            util.DEFAULT_ACTIVATION,
            total_steps,
            util.DEFAULT_MAX_SEQ_LEN,
            schedule=args.schedule,
            dropout=util.DEFAULT_DROPOUT,
            warm_up_steps=args.warm_up_steps,
            **extra_args
        )
        model.type_token_fc = nn.Linear(args.d_model, 10)
        print("random model")
    elif args.model_type == "unified":
        model = UnifiedModel(
            sampler,
            pad_token_idx,
            vocab_size,
            args.d_model,
            args.num_layers,
            args.num_heads,
            args.d_feedforward,
            args.lr,
            DEFAULT_WEIGHT_DECAY,
            util.DEFAULT_ACTIVATION,
            total_steps,
            util.DEFAULT_MAX_SEQ_LEN,
            schedule=args.schedule,
            dropout=util.DEFAULT_DROPOUT,
            warm_up_steps=args.warm_up_steps,
            **extra_args
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    return model


def load_model(args, sampler, vocab_size, total_steps, pad_token_idx):
    # These args don't affect the model directly but will be saved by lightning as hparams
    # Tensorboard doesn't like None so we need to convert to string
    train_tokens = "None" if args.train_tokens is None else args.train_tokens
    num_buckets = "None" if args.num_buckets is None else args.num_buckets
    extra_args = {
        "batch_size": args.batch_size,
        "acc_batches": args.acc_batches,
        "epochs": args.epochs,
        "clip_grad": args.clip_grad,
        "augment": args.augment,
        "aug_prob": args.aug_prob,
        "train_tokens": train_tokens,
        "num_buckets": num_buckets,
        "limit_val_batches": args.limit_val_batches
    }

    # If no model is given, use random init
    if args.model_path in ["none", "None"]:
        model = load_rand_model(args, extra_args, sampler, vocab_size, total_steps, pad_token_idx)
    else:
        if args.model_type == "bart":
            print("123")
            checkpoint = pl_load(args.model_path, map_location=lambda storage, loc: storage)
            model = BARTModel(
                        sampler,
                        pad_token_idx,
                        vocab_size,
                        args.d_model,
                        args.num_layers,
                        args.num_heads,
                        args.d_feedforward,
                        args.lr,
                        DEFAULT_WEIGHT_DECAY,
                        util.DEFAULT_ACTIVATION,
                        total_steps,
                        util.DEFAULT_MAX_SEQ_LEN,
                        schedule=args.schedule,
                        dropout=util.DEFAULT_DROPOUT,
                        warm_up_steps=args.warm_up_steps,
                        **extra_args
                    )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.type_token_fc = nn.Linear(args.d_model, 10)

            # prompt
            # vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 88, 256, 512, 8
            # norm_shape, ffn_num_input, num_layers, dropout = [256], 256, 3, 0.2
            # model.type_token_fc = nn.Linear(args.d_model, 10)

            # num_hiddens = 256
            # model.prompt_model = TPrompt(args.d_model, num_hiddens, args.num_heads ,args.num_layers, 3, vocab_size=vocab_size, n_prefix_conv=64)
            # model.prompt_model.gcn_model.load_state_dict(torch.load('gcn_3layer.pt'))

            # model.prompt_model.load_state_dict(torch.load('fuse_pretrain_mask_3layer.pt'), strict=False)
            # if args.reaction_model_path is not None:
            #     checkpoint = pl_load(args.reaction_model_path, map_location=lambda storage, loc: storage)
            #     model.reaction_type_model = ReactionTypeModel(
            #                 sampler,
            #                 pad_token_idx,
            #                 vocab_size,
            #                 args.d_model,
            #                 args.num_layers,
            #                 args.num_heads,
            #                 args.d_feedforward,
            #                 args.lr,
            #                 DEFAULT_WEIGHT_DECAY,
            #                 util.DEFAULT_ACTIVATION,
            #                 total_steps,
            #                 util.DEFAULT_MAX_SEQ_LEN,
            #                 schedule=args.schedule,
            #                 dropout=util.DEFAULT_DROPOUT,
            #                 warm_up_steps=args.warm_up_steps,
            #                 **extra_args
            #             )
            #     model.reaction_type_model.load_state_dict(checkpoint["state_dict"], strict=True)


            # for name, parameters in model.prompt_model.graph_model.named_parameters():
            #     parameters.requires_grad = False
                
            # if model.emb.weight.requires_grad:
            #     # print("emb forward", self.emb.weight.requires_grad)
            #     model.emb.weight.requires_grad = False
            #     for name, parameters in model.encoder.named_parameters():
            #         parameters.requires_grad = False
            #     for name, parameters in model.decoder.named_parameters():
            #         parameters.requires_grad = False
            #     model.token_fc.weight.requires_grad = False
        
        elif args.model_type == "reactionType":
            checkpoint = pl_load(args.model_path, map_location=lambda storage, loc: storage)
            model = ReactionTypeModel(
                        sampler,
                        pad_token_idx,
                        vocab_size,
                        args.d_model,
                        # args.num_layers,
                        6,
                        args.num_heads,
                        args.d_feedforward,
                        args.lr,
                        DEFAULT_WEIGHT_DECAY,
                        util.DEFAULT_ACTIVATION,
                        total_steps,
                        util.DEFAULT_MAX_SEQ_LEN,
                        schedule=args.schedule,
                        dropout=util.DEFAULT_DROPOUT,
                        warm_up_steps=args.warm_up_steps,
                        **extra_args
                    )
            model.load_state_dict(checkpoint["state_dict"], strict=False)

            model.type_token_fc = nn.Linear(args.d_model, 10)

            # prompt
            # vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 88, 256, 512, 8
            # norm_shape, ffn_num_input, num_layers, dropout = [256], 256, 3, 0.2
            # num_hiddens = 256
            # model.prompt_model = TPrompt(args.d_model, num_hiddens, args.num_heads ,args.num_layers, 3, vocab_size=vocab_size, n_prefix_conv=64)
            # model.prompt_model.gcn_model.load_state_dict(torch.load('gcn_3layer.pt'))

            # model.load_state_dict(torch.load('fuse_pretrain_mask_3layer.pt'), strict=False)

            # for name, parameters in model.prompt_model.graph_model.named_parameters():
            #     parameters.requires_grad = False

        elif args.model_type == "unified":
            model = UnifiedModel.load_from_checkpoint(
                args.model_path,
                decode_sampler=sampler,
                pad_token_idx=pad_token_idx,
                vocab_size=vocab_size,
                num_steps=total_steps,
                lr=args.lr,
                weight_decay=args.weight_decay,
                schedule=args.schedule,
                warm_up_steps=args.warm_up_steps,
                **extra_args
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

    return model


def main(args):
    # util.seed_everything(73)

    if args.task in ["forward_prediction", "mol_opt"]:
        forward_pred = True
    elif args.task == "backward_prediction":
        forward_pred = False
    else:
        raise ValueError(f"Unknown task {args.task}")
    print(args.data_path)
    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = util.build_dataset(args, forward=forward_pred)
    print("Finished dataset.")

    print("Building data module...")
    dm = util.build_reaction_datamodule(args, dataset, tokeniser, forward=forward_pred)
    num_available_cpus = len(os.sched_getaffinity(0))
    num_workers = num_available_cpus // args.gpus
    dm._num_workers = num_workers
    print(f"Using {str(num_workers)} workers for data module.")
    print("Finished datamodule.")

    vocab_size = len(tokeniser)
    train_steps = util.calc_train_steps(args, dm)
    print(f"Train steps: {train_steps}")

    sampler = DecodeSampler(tokeniser, util.DEFAULT_MAX_SEQ_LEN)
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

    print("Loading model...")
    model = load_model(args, sampler, vocab_size, train_steps + 1, pad_token_idx)
    sampler.max_seq_len = model.max_seq_len
    print("Finished model.")

    print("Building trainer...")
    trainer = util.build_trainer(args)
    print("Finished trainer.")

    print("Fitting data module to model")
    print("pre model paras", model.emb.weight.requires_grad)
    trainer.fit(model, datamodule=dm)
    print("last model paras", model.emb.weight.requires_grad)
    print("Finished training.")

    print("Printing unknown tokens...")
    tokeniser.print_unknown_tokens()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path", type=str, default="data/uspto_50.pickle")
    parser.add_argument("--task", type=str, default="backward_prediction")
    parser.add_argument("--model_path", type=str, default="models/combined/step=1000000.ckpt")
    parser.add_argument("--dataset", type=str, default="uspto_50")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--vocab_path", type=str, default=DEFAULT_DOWNSTREAM_VOCAB_PATH)
    parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)
    parser.add_argument("--log_dir", type=str, default=util.DEFAULT_LOG_DIR)
    parser.add_argument("--deepspeed_config_path", type=str, default=util.DEFAULT_DEEPSPEED_CONFIG_PATH)

    # Model and training args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--augment", type=str, default=DEFAULT_AUGMENT)
    parser.add_argument("--aug_prob", type=float, default=DEFAULT_AUG_PROB)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--num_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument("--limit_val_batches", type=float, default=DEFAULT_LIMIT_VAL_BATCHES)
    parser.add_argument("--gpus", type=int, default=util.DEFAULT_GPUS)
    parser.add_argument("--num_nodes", type=int, default=util.DEFAULT_NUM_NODES)

    # Rand init model args
    parser.add_argument("--d_model", type=int, default=util.DEFAULT_D_MODEL)
    parser.add_argument("--num_layers", type=int, default=util.DEFAULT_NUM_LAYERS)
    parser.add_argument("--num_heads", type=int, default=util.DEFAULT_NUM_HEADS)
    parser.add_argument("--d_feedforward", type=int, default=util.DEFAULT_D_FEEDFORWARD)

    # model for reaction_type
    parser.add_argument("--reaction_model_path", type=str, default=None)

    args = parser.parse_args()
    main(args)
