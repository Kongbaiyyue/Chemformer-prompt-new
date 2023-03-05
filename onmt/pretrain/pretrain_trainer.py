import traceback

import torch
import torch.nn as nn

import onmt
from onmt.pretrain import build_loss_compute
from onmt.utils.logging import logger


def build_trainer(opt, device_id, Bert_model, GNN_model, fields, optim_Bert, optim_GNN, model_saver_Bert=None, model_saver_GNN=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    # train_loss_Bert = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
    # train_loss_GNN = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
    train_loss_Bert = build_loss_compute()
    train_loss_GNN = build_loss_compute()
    latent_loss = None
    # if model.n_latent > 1:
    #     latent_loss = onmt.utils.loss.build_loss_compute(
    #         model, tgt_field, opt, train=False, reduce=False)
    # valid_loss = onmt.utils.loss.build_loss_compute(
    #     model, tgt_field, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    # earlystopper = onmt.utils.EarlyStopping(
    #     opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
    #     if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = PreTrainer(Bert_model, GNN_model, train_loss_Bert, train_loss_GNN, optim_Bert, optim_GNN, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver_Bert=model_saver_Bert if gpu_rank == 0 else None,
                           model_saver_GNN=model_saver_GNN if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=None,
                           dropout=dropout,
                           dropout_steps=dropout_steps,
                           latent_loss=latent_loss,
                           segment_token_idx=None,
                           stoi=tgt_field.vocab.stoi,
                           max_dist=opt.max_distance,
                           alpha=opt.alpha)
    return trainer


class PreTrainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model_Bert, model_GNN, train_loss_Bert, train_loss_GNN, optim_Bert, optim_GNN,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver_Bert=None, model_saver_GNN=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0],
                 latent_loss=None, segment_token_idx=None, stoi=None, max_dist=None,
                 alpha=1.0):
        # Basic attributes.
        self.model_Bert = model_Bert
        self.model_GNN = model_GNN
        self.train_loss_Bert = train_loss_Bert
        self.train_loss_GNN = train_loss_GNN
        self.optim_Bert = optim_Bert
        self.optim_GNN = optim_GNN
        self.latent_loss = latent_loss
        # self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver_Bert = model_saver_Bert
        self.model_saver_GNN = model_saver_GNN
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.segment_token_idx = segment_token_idx
        self.stoi = stoi
        self.inv_stoi = {v: k for k, v in stoi.items()}
        self.max_dist = max_dist
        self.alpha = alpha
        self.train_Bert = False
        self.train_GNN = False

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        # self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    # def _maybe_update_dropout(self, step):
    #     for i in range(len(self.dropout_steps)):
    #         if step > 1 and step == self.dropout_steps[i] + 1:
    #             self.model.update_dropout(self.dropout[i])
    #             logger.info("Updated dropout to %f from step %d"
    #                         % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim_Bert.training_step)
        for batch in iterator:
            batches.append(batch)

            normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim_Bert.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    # def _update_average(self, step):
    #     if self.moving_average is None:
    #         copy_params = [params.detach().float()
    #                        for params in self.model.parameters()]
    #         self.moving_average = copy_params
    #     else:
    #         average_decay = max(self.average_decay,
    #                             1 - (step + 1)/(step + 10))
    #         for (i, avg), cpt in zip(enumerate(self.moving_average),
    #                                  self.model.parameters()):
    #             self.moving_average[i] = \
    #                 (1 - average_decay) * avg + \
    #                 cpt.detach().float() * average_decay

    def train(self,
              train_iter,
              fuse_train_steps,
              GNN_train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        if self.gpu_rank is not None:
            device = torch.device("cuda", self.gpu_rank)
        else:
            device = torch.device("cpu")

        self.train_GNN = True
        self.train_Bert = False

        GNN_train_steps = 1

        t = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32, device=device))

        j_step = 0
        
        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            GNN_step = self.optim_GNN.training_step
            Bert_step = self.optim_Bert.training_step

            if j_step >= 100:
                j_step = 0
                if GNN_step < GNN_train_steps:
                    self.train_Bert = False
                    self.train_GNN = True
                elif self.train_GNN:
                    self.train_GNN = False
                    self.train_Bert = True
                elif self.train_Bert:
                    self.train_Bert = False
                    self.train_GNN = True
            j_step += 1

            if Bert_step > fuse_train_steps and fuse_train_steps > 0:
                break

            for k, batch in enumerate(batches):
                target_size = batch.tgt.size(0)
                batch_size = batch.batch_size
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)

                if self.train_Bert:
                    with torch.no_grad():
                        node_emb, edge_emb = self.model_GNN(batch.atom[0].to(torch.float32),
                                                            batch.edge[0].to(torch.float32), lengths=batch.atom[1][0],
                                                            adj=batch.edge_adj[0])
                        target = node_emb[:, 0, :]
                        target = nn.functional.normalize(target, p=2, dim=1)
                    src_new = src.squeeze(-1)
                    global_token = torch.full((1, batch_size), 88, device=device)
                    src_new = torch.cat((global_token, src_new), 0)
                    output = self.model_Bert(src_new, None, src_lengths)[:, 0, :]
                    output = nn.functional.normalize(output, p=2, dim=1)

                    logits = target @ output.transpose(1, 0) * torch.exp(t)

                    # logits_t = output @ target.transpose(1, 0) * torch.exp(t)
                    # logits_g = target @ output.transpose(1, 0) * torch.exp(t)

                    labels = torch.arange(batch_size, device=device)

                    try:
                        loss = self.train_loss_Bert(
                            logits,
                            labels)
                        # loss_t = self.train_loss_Bert(
                        #     logits_t,
                        #     labels)
                        # loss_g = self.train_loss_Bert(
                        #     logits_g,
                        #     labels)
                        # loss = (loss_t + loss_g) / 2

                        if loss is not None:
                            self.optim_Bert.backward(loss)

                    except Exception:
                        traceback.print_exc()
                        logger.info("At step %d, we removed a batch - accum %d",
                                    self.optim_Bert.training_step, k)

                    self.optim_Bert.step()

                elif self.train_GNN:
                    with torch.no_grad():
                        src_new = src.squeeze(-1)
                        global_token = torch.full((1, batch_size), 88, device=device)
                        src_new = torch.cat((global_token, src_new), 0)
                        target = self.model_Bert(src_new, None, src_lengths)[:, 0, :]
                        target = nn.functional.normalize(target, p=2, dim=1)
                    
                    node_emb, edge_emb = self.model_GNN(batch.atom[0].to(torch.float32), batch.edge[0].to(torch.float32), lengths=batch.atom[1][0], adj=batch.edge_adj[0])
                    output = node_emb[:, 0, :]
                    output = nn.functional.normalize(output, p=2, dim=1)

                    # logits_t = target @ output.transpose(1, 0) * torch.exp(t)
                    # logits_g = output @ target.transpose(1, 0) * torch.exp(t)
                    logits = target @ output.transpose(1, 0) * torch.exp(t)

                    labels = torch.arange(batch_size, device=device)

                    try:
                        loss = self.train_loss_GNN(
                                logits,
                                labels)
                        # loss_t = self.train_loss_GNN(
                        #     logits_t,
                        #     labels)
                        # loss_g = self.train_loss_GNN(
                        #     logits_g,
                        #     labels)
                        # loss = (loss_t + loss_g) / 2

                        if loss is not None:
                            self.optim_GNN.backward(loss)

                    except Exception:
                        traceback.print_exc()
                        logger.info("At step %d, we removed a batch - accum %d",
                                    self.optim_GNN.training_step, k)

                    self.optim_GNN.step()
                # print(loss)
            if GNN_step % 1000 == 0:
                print(loss)
                print("lr :", self.optim_Bert._learning_rate)

        self.model_saver_Bert.save_by_name(50000, model_name="model_Bert")
        self.model_saver_GNN.save_by_name(50000, model_name="model_GNN")

