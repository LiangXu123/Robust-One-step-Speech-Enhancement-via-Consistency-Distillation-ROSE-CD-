import time
from math import ceil
import warnings

import torch
import sys
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample
from .nn import mean_flat, append_dims, append_zero
from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss
import copy
import torch.nn as nn
import numpy as np
import torch
from torchmetrics import ScaleInvariantSignalDistortionRatio
from asteroid.losses import pairwise_neg_sisdr
from asteroid.losses import PITLossWrapper

# target = torch.tensor([3.0, -0.5, 2.0, 7.0])
# preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
# si_sdr = ScaleInvariantSignalDistortionRatio()
# si_sdr(preds, target)


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999,
                            help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03,
                            help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20,
                            help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="score_matching",
                            help="The type of loss function to use.[score_matching, denoiser, data_prediction]")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2",
                            help="The weighting of the loss function.")
        parser.add_argument("--network_scaling", type=str,
                            default=None, help="The type of loss scaling to use.")
        parser.add_argument("--c_in", type=str, default="1",
                            help="The input scaling for x.")
        parser.add_argument("--c_out", type=str, default="1",
                            help="The output scaling.")
        parser.add_argument("--c_skip", type=str, default="0",
                            help="The skip connection scaling.")
        parser.add_argument("--sigma_data", type=float,
                            default=0.1, help="The data standard deviation.")
        parser.add_argument("--l1_weight", type=float, default=0.0,
                            help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--consistency_weight", type=float, default=1.0,
                            help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--pesq_weight", type=float, default=0.0,
                            help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--sisdr_weight", type=float, default=0.0,
                            help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--sr", type=int, default=16000,
                            help="The sample rate of the audio files.")

        # parser.add_argument("--use_distill_mode", type=str,
        #                     default=False, help="Set the model to distill mode.")
        # parser.add_argument("--distill_loss_type", type=str,
        # default='L2', help="distill loss between model,target_model.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.9999, t_eps=0.03, num_eval_files=20, loss_type='score_matching',
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1,
        l1_weight=0.001, pesq_weight=0.0, sisdr_weight=0.0, consistency_weight=1.0, sr=16000, data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae', [score_matching, denoiser, data_prediction]
        """
        super().__init__()
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        # self.master_paras = self.parameters()
        # Filter parameters to exclude those related to target_score_model or teacher_score_model
        self.master_paras = [param for name, param in self.named_parameters()
                             if "target_score_model" not in name and "teacher_score_model" not in name]

        # self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self.ema = ExponentialMovingAverage(
            self.master_paras, decay=self.ema_decay)

        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.l1_weight = l1_weight
        self.pesq_weight = pesq_weight
        self.sisdr_weight = sisdr_weight
        self.consistency_weight = consistency_weight
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.sr = sr
        # Initialize PESQ loss if pesq_weight > 0.0
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(
            **kwargs, gpu=kwargs.get('gpus', 0) > 0)

        self.use_distill_mode = False
        self.distill_sampler_already_init = False
        self.sisdr_already_init = False
        self.distill_solver = None
        self.distill_sample_return_xmean = False
        self.with_pesq_loss = False
        self.with_sisdr_loss = False
        self.distill_loss_type = None
        self.sisdr_loss_func = None

    def get_pc_sampler2(self, predictor_name='reverse_diffusion', corrector_name='ald',
                        score_fn=None, denoise=True, corrector_steps=1, snr=0.5):
        # Reverse sampling
        return sampling.get_pc_sampler2(
            predictor_name, corrector_name, self.sde, score_fn,
            denoise=denoise, eps=self.t_eps, snr=snr, corrector_steps=corrector_steps
        )

    def get_heun_solver_sampler(self, predictor_name='heun_solver',
                                score_fn=None, denoise=True):
        # Reverse sampling
        return sampling.get_heun_solver_sampler(
            predictor_name, self.sde, score_fn,
            denoise=denoise, eps=self.t_eps)

    def get_mix_sampler(self, predictor_name='mix_solver',
                        score_fn=None, denoise=True):
        # Reverse sampling
        return sampling.get_mix_sampler(
            predictor_name, self.sde, score_fn,
            denoise=denoise, eps=self.t_eps)

    def get_euler_sampler(self, predictor_name='euler_maruyama',
                          score_fn=None, denoise=True):
        # Reverse sampling
        return sampling.get_euler_sampler(
            predictor_name, self.sde, score_fn,
            denoise=denoise, eps=self.t_eps)

    def get_sb_solver(self, score_fn=None, denoise=True):
        # Reverse sampling
        if self.distill_solver == 'sb':
            return sampling.get_sb_solver(
                self.sde, score_fn,
                denoise=denoise, eps=self.t_eps)
        elif self.distill_solver == 'sb_sde':
            return sampling.get_sb_solver_sde(
                self.sde, score_fn,
                denoise=denoise, eps=self.t_eps)

    def get_stochastic_sampler(self, score_fn, y, N, snr=0.5):
        # Reverse sampling for consistency distillation
        sde = self.sde.copy()
        sde.N = N
        return sampling.get_stochastic_sampler(score_fn, sde,
                                               y, snr)

    def distill_on(self, distill_solver='pc2', distill_loss_type="L2", with_pesq_loss=False, with_sisdr_loss=False, distill_sample_return_xmean=False,
                   main_device="cuda:1", target_device="cuda:2", teach_device="cuda:3", corrector_steps=1, snr=0.5,
                   weight_schedule="uniform", sigma_data=0.1):  #
        self.use_distill_mode = True
        self.distill_solver = distill_solver
        self.distill_loss_type = distill_loss_type
        self.distill_sample_return_xmean = distill_sample_return_xmean
        self.corrector_steps = corrector_steps
        self.snr = snr
        self.weight_schedule = weight_schedule
        self.sigma_data = sigma_data
        assert self.loss_type == "precond_denoiser"
        self.with_pesq_loss = with_pesq_loss
        self.with_sisdr_loss = with_sisdr_loss
        print("pesq_weight:", self.pesq_weight)
        print("sisdr_weight:", self.sisdr_weight)
        print("consistency_weight:", self.consistency_weight)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.master_paras, lr=self.lr)
        # optimizer = torch.optim.RAdam(self.master_paras, lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())
        if self.use_distill_mode:
            # Copy EMA parameters to target_score_model
            self.ema.copy_to(self.target_score_model.dnn.parameters())

    def on_load_checkpoint(self, checkpoint):
        # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
        # return True
        ema = checkpoint.get('ema', None)
        # print('ema:', checkpoint['ema'])
        # print('self.master_paras:', self.master_paras)
        # print('self.named_parameters():', self.named_parameters())
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        if self.use_distill_mode:
            """ Remove layers containing 'teacher_score_model' or 'target_score_model' before saving """
            keys_to_remove = [key for key in checkpoint["state_dict"]
                              if "teacher_score_model" in key or "target_score_model" in key]
            for key in keys_to_remove:
                del checkpoint["state_dict"][key]
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode=True, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                # store current params in EMA
                self.ema.store(self.dnn.parameters())
                # copy EMA parameters over current params for evaluation
                self.ema.copy_to(self.dnn.parameters())
            else:
                # train
                # pass
                if self.ema.collected_params is not None:
                    # restore the EMA weights (if stored)
                    self.ema.restore(self.dnn.parameters())
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def make_distill_loss(self, distiller, distiller_target, t):
        """
        Different loss functions can be used to train the score model, see the paper:

        Julius Richter, Danilo de Oliveira, and Timo Gerkmann
        "Investigating Training Objectives for Generative Speech Enhancement"
        https://arxiv.org/abs/2409.10753

        """
        def get_weightings(weight_schedule, snrs, sigma_data):
            if weight_schedule == "snr":
                weightings = snrs
            elif weight_schedule == "snr+1":
                weightings = snrs + 1
            elif weight_schedule == "karras":
                weightings = snrs + 1.0 / sigma_data**2
            elif weight_schedule == "truncated-snr":
                weightings = torch.clamp(snrs, min=1.0)
            elif weight_schedule == "uniform":
                weightings = torch.ones_like(snrs)
            else:
                raise NotImplementedError()
            return weightings

        snrs = t**-2
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        # weights = 1.0
        if self.distill_loss_type == "L2":
            # diffs = (distiller - distiller_target) ** 2 # this will return complex as well
            diffs = torch.square(
                # Eq. (7), the abs will convert x+yj to real number only
                torch.abs(distiller - distiller_target))
            loss = mean_flat(diffs) * weights
        else:
            raise ValueError("Invalid loss type: {}".format(
                self.distill_loss_type))
        return loss.mean(), weights

    def make_pesq_loss(self, x_hat, x):
        B, C, F, T = x.shape
        assert self.pesq_weight > 0.0
        if self.pesq_weight < 1e-7:
            return 0
        # calculate pesq loss, just like sb did
        # losses in the time domain (td)
        target_len = (self.data_module.num_frames - 1) * \
            self.data_module.hop_length
        x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
        x_td = self.to_audio(x.squeeze(), target_len)
        # losses using PESQ
        losses_pesq = self.pesq_loss(x_td, x_hat_td)
        losses_pesq = torch.mean(losses_pesq)
        losses_pesq = self.pesq_weight * losses_pesq
        return losses_pesq

    def make_sisdr_loss(self, x_hat, x):
        assert self.sisdr_weight > 0.0
        if not self.sisdr_already_init:
            self.sisdr_already_init = True
            self.sisdr_loss_func = PITLossWrapper(pairwise_neg_sisdr,
                                                  pit_from='pw_mtx')
        # calculate pesq loss, just like sb did
        # losses in the time domain (td)
        target_len = (self.data_module.num_frames - 1) * \
            self.data_module.hop_length
        x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
        x_td = self.to_audio(x.squeeze(), target_len)
        # losses using sisdr
        losses_sisdr = self.sisdr_loss_func(
            x_hat_td.unsqueeze(1), x_td.unsqueeze(1))
        losses_sisdr = self.sisdr_weight * losses_sisdr
        return losses_sisdr

    def _distill_step(self, batch, batch_idx):
        # batch, target_score_model, teacher_score_model = batch
        # Use models attached in `CustomCallback`
        # to match the same name's with main model
        x, y = batch

        def denoise_fn(score, x_t, t, sigma=None):
            # convert the score from dnn model to denoiser output
            # in edm ,they use skip connection:# return skip connect result = c_out * model_output + c_skip * x_t
            # but in sgmse, we do just convert score to D like Eq. (10)
            if sigma == None:
                sigma = self.sde._std(t)[:, None, None, None]
            # score = forward_out
            D = score * sigma.pow(2) + x_t  # equivalent to Eq. (10)
            return D

        @torch.no_grad()
        def target_denoise_fn(score, x_t, t, sigma=None):
            # convert the score from dnn model to denoiser output
            # in edm ,they use skip connection:# return skip connect result = c_out * model_output + c_skip * x_t
            # but in sgmse, we do just convert score to D like Eq. (10)
            if sigma == None:
                sigma = self.sde._std(t)[:, None, None, None]
            # score = forward_out
            D = score * sigma.pow(2) + x_t  # equivalent to Eq. (10)
            return D

        # self.sde.T = 1
        # self.sde.N = 30
        timesteps = torch.linspace(
            self.sde.T, self.t_eps, self.sde.N, device=x.device)
        indices = torch.randint(
            0, self.sde.N - 1, (x.shape[0],)
            # , device=x.device
        )
        t = timesteps[indices]
        t2 = timesteps[indices + 1]
        # print('t:{}, t2:{}'.format(t, t2))
        # t:tensor([0.9331, 0.5652, 0.4648, 0.9331, 0.5986, 1.0000, 0.8997, 0.1638])
        # t2:tensor([0.8997, 0.5317, 0.4314, 0.8997, 0.5652, 0.9666, 0.8662, 0.1303])
        mean, std = self.sde.marginal_prob(x, y, t)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]
        # sigmat2 = self.sde._std(t2)[:, None, None, None]
        x_t = mean + sigma * z

        forward_out = self(x_t, y, t)
        distiller = denoise_fn(forward_out, x_t, t, sigma)

        @torch.no_grad()
        def batch_sampler(sampler_fn, xt, y, t, next_t):
            # loop for bs
            samples = []
            for bs_i in range(len(xt)):
                sample = sampler_fn(
                    xt[bs_i:bs_i+1, :], y[bs_i:bs_i+1, :], t[bs_i], next_t[bs_i])
                samples.append(sample)
            samples = torch.cat(samples, dim=0)
            return samples

        @torch.no_grad()
        def pc2_solver(xt, y, t, next_t):
            if not self.distill_sampler_already_init:
                self.distill_sampler_already_init = True
                self.distill_sampler = self.get_pc_sampler2(
                    'reverse_diffusion_with_dnn', 'ald_with_dnn', score_fn=self.teacher_score_model,
                    denoise=self.distill_sample_return_xmean, corrector_steps=self.corrector_steps, snr=self.snr)
                # return pc_sampler_at_one_step, but this is only for bs=1, but in train we have bs=8
            return batch_sampler(self.distill_sampler, xt, y, t, next_t)

        @torch.no_grad()
        def heun_solver(xt, y, t, next_t):
            if not self.distill_sampler_already_init:
                self.distill_sampler_already_init = True
                self.distill_sampler = self.get_heun_solver_sampler(
                    'heun_solver', score_fn=self.teacher_score_model,
                    denoise=self.distill_sample_return_xmean)
                # return pc_sampler_at_one_step, but this is only for bs=1, but in train we have bs=8
            return batch_sampler(self.distill_sampler, xt, y, t, next_t)

        @torch.no_grad()
        def mix_solver(xt, y, t, next_t):
            if not self.distill_sampler_already_init:
                self.distill_sampler_already_init = True
                self.distill_sampler = self.get_mix_sampler(
                    'mix_solver', score_fn=self.teacher_score_model,
                    denoise=self.distill_sample_return_xmean)
                # return pc_sampler_at_one_step, but this is only for bs=1, but in train we have bs=8
            return batch_sampler(self.distill_sampler, xt, y, t, next_t)

        @torch.no_grad()
        def euler_solver(xt, y, t, next_t):
            if not self.distill_sampler_already_init:
                self.distill_sampler_already_init = True
                self.distill_sampler = self.get_euler_sampler(
                    'euler_maruyama', score_fn=self.teacher_score_model,
                    denoise=self.distill_sample_return_xmean)
                # return pc_sampler_at_one_step, but this is only for bs=1, but in train we have bs=8
            return batch_sampler(self.distill_sampler, xt, y, t, next_t)

        @torch.no_grad()
        def sb_solver(xt, y, t, next_t):
            if not self.distill_sampler_already_init:
                self.distill_sampler_already_init = True
                self.distill_sampler = self.get_sb_solver(
                    score_fn=self.teacher_score_model, denoise=self.distill_sample_return_xmean)
                # return pc_sampler_at_one_step, but this is only for bs=1, but in train we have bs=8
            return batch_sampler(self.distill_sampler, xt, y, t, next_t)

        # calculate x_t+1 here
        if self.distill_solver == 'pc2':
            targt_x_t2 = pc2_solver(
                x_t, y, t, t2).detach()
        # calculate x_t+1 here
        elif self.distill_solver == 'heun':
            targt_x_t2 = heun_solver(
                x_t, y, t, t2).detach()
        # calculate x_t+1 here
        elif self.distill_solver == 'mix':
            targt_x_t2 = mix_solver(
                x_t, y, t, t2).detach()
        # calculate x_t+1 here
        elif self.distill_solver == 'euler':
            targt_x_t2 = euler_solver(
                x_t, y, t, t2).detach()
        # calculate x_t+1 here
        elif self.distill_solver == 'sb' or self.distill_solver == 'sb_sde':
            targt_x_t2 = sb_solver(
                x_t, y, t, t2).detach()
        else:
            raise ValueError(
                'Your {} is not supported!'.format(self.distill_solver))

        # forward x_t+1 to target model, to get the distill target
        # target_forward_out = self.target_score_model(
        #     targt_x_t2, y, t2).detach()
        # distiller_target = target_denoise_fn(
        #     target_forward_out, targt_x_t2, t2, sigma=sigmat2).detach()
        distiller_target = self.target_score_model(
            targt_x_t2, y, t2, return_X=True).detach()

        loss, loss_weights = self.make_distill_loss(
            distiller, distiller_target, t)

        loss_dict = dict()
        loss_dict['Consistency_loss'] = loss * self.consistency_weight
        train_loss = loss * self.consistency_weight

        if self.with_pesq_loss:
            pesq_loss = self.make_pesq_loss(distiller, x)
            loss_dict['PESQ_loss'] = pesq_loss
            train_loss += pesq_loss
        else:
            loss_dict['PESQ_loss'] = 0.

        if self.with_sisdr_loss:
            sisdr_loss = self.make_sisdr_loss(distiller, x)
            loss_dict['SISDR_loss'] = sisdr_loss
            train_loss += sisdr_loss
        else:
            loss_dict['SISDR_loss'] = 0.

        loss_dict['loss'] = train_loss

        return loss_dict

    def _loss(self, forward_out, x_t, z, t, mean, x):
        """
        Different loss functions can be used to train the score model, see the paper:

        Julius Richter, Danilo de Oliveira, and Timo Gerkmann
        "Investigating Training Objectives for Generative Speech Enhancement"
        https://arxiv.org/abs/2409.10753

        """

        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            score = forward_out
            if self.loss_weighting == "sigma^2":
                losses = torch.square(torch.abs(score * sigma + z))  # Eq. (7)
            else:
                raise ValueError("Invalid loss weighting for loss_type=score_matching: {}".format(
                    self.loss_weighting))
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(
                0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        elif self.loss_type == "denoiser" or self.loss_type == "precond_denoiser":
            score = forward_out
            D = score * sigma.pow(2) + x_t  # equivalent to Eq. (10)
            losses = torch.square(torch.abs(D - mean))  # Eq. (8)
            if self.loss_weighting == "1":
                losses = losses
            elif self.loss_weighting == "sigma^2":
                losses = losses * sigma**2
            elif self.loss_weighting == "edm":
                losses = ((sigma**2 + self.sigma_data**2) /
                          ((sigma*self.sigma_data)**2))[:, None, None, None] * losses
            else:
                raise ValueError(
                    "Invalid loss weighting for loss_type=denoiser: {}".format(self.loss_weighting))
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(
                0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        elif self.loss_type == "data_prediction":
            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = (1/(F*T))*torch.square(torch.abs(x_hat - x))
            losses_tf = torch.mean(
                0.5*torch.sum(losses_tf.reshape(losses_tf.shape[0], -1), dim=-1))

            # losses in the time domain (td)
            target_len = (self.data_module.num_frames - 1) * \
                self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = torch.mean(
                0.5*torch.sum(losses_l1.reshape(losses_l1.shape[0], -1), dim=-1))

            # losses using PESQ
            if self.pesq_weight > 0.0:
                losses_pesq = self.pesq_loss(x_td, x_hat_td)
                losses_pesq = torch.mean(losses_pesq)
                # combine the losses
                loss = losses_tf + self.l1_weight * losses_l1 + self.pesq_weight * losses_pesq
            else:
                loss = losses_tf + self.l1_weight * losses_l1
        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_type))

        return loss

    def _step(self, batch, batch_idx):
        x, y = batch
        t = torch.rand(x.shape[0], device=x.device) * \
            (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, y, t)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        forward_out = self(x_t, y, t)
        loss = self._loss(forward_out, x_t, z, t, mean, x)
        return loss

    def training_step(self, batch, batch_idx):
        if self.use_distill_mode:
            loss = self._distill_step(batch, batch_idx)
        else:
            loss = self._step(batch, batch_idx)
        if isinstance(loss, dict):
            self.log('train_loss', loss['loss'], on_step=True,
                     on_epoch=True, sync_dist=True, prog_bar=True)
            self.log('Consistency_loss', loss['Consistency_loss'], on_step=True,
                     on_epoch=True, sync_dist=True, prog_bar=True)
            self.log('PESQ_loss', loss['PESQ_loss'], on_step=True,
                     on_epoch=True, sync_dist=True, prog_bar=True)
            self.log('SISDR_loss', loss['SISDR_loss'], on_step=True,
                     on_epoch=True, sync_dist=True, prog_bar=True)
        else:
            self.log('train_loss', loss, on_step=True,
                     on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            rank = self.trainer.global_rank  # Lightning's built-in rank
            world_size = self.trainer.world_size

            # Split the evaluation files among the GPUs
            eval_files_per_gpu = self.num_eval_files // world_size

            clean_files = self.data_module.valid_set.clean_files[:self.num_eval_files]
            noisy_files = self.data_module.valid_set.noisy_files[:self.num_eval_files]

            # Select the files for this GPU
            if rank == world_size - 1:
                clean_files = clean_files[rank*eval_files_per_gpu:]
                noisy_files = noisy_files[rank*eval_files_per_gpu:]
            else:
                clean_files = clean_files[rank *
                                          eval_files_per_gpu:(rank+1)*eval_files_per_gpu]
                noisy_files = noisy_files[rank *
                                          eval_files_per_gpu:(rank+1)*eval_files_per_gpu]

            # Evaluate the performance of the model
            pesq_sum = 0
            si_sdr_sum = 0
            estoi_sum = 0
            for (clean_file, noisy_file) in zip(clean_files, noisy_files):
                # Load the clean and noisy speech
                x, sr_x = load(clean_file)
                x = x.squeeze().numpy()
                y, sr_y = load(noisy_file)
                assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"

                # Resample if necessary
                if sr_x != 16000:
                    x_16k = resample(
                        x, orig_sr=sr_x, target_sr=16000).squeeze()
                else:
                    x_16k = x

                # Enhance the noisy speech
                x_hat = self.enhance(y, N=self.sde.N)
                if self.sr != 16000:
                    x_hat_16k = resample(
                        x_hat, orig_sr=self.sr, target_sr=16000).squeeze()
                else:
                    x_hat_16k = x_hat

                pesq_sum += pesq(16000, x_16k, x_hat_16k, 'wb')
                si_sdr_sum += si_sdr(x, x_hat)
                estoi_sum += stoi(x, x_hat, self.sr, extended=True)

            pesq_avg = pesq_sum / len(clean_files)
            si_sdr_avg = si_sdr_sum / len(clean_files)
            estoi_avg = estoi_sum / len(clean_files)

            self.log('pesq', pesq_avg, on_step=False,
                     on_epoch=True, sync_dist=True)
            self.log('si_sdr', si_sdr_avg, on_step=False,
                     on_epoch=True, sync_dist=True)
            self.log('estoi', estoi_avg, on_step=False,
                     on_epoch=True, sync_dist=True)

        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False,
                 on_epoch=True, sync_dist=True)

        return loss

    def forward(self, x_t, y, t, return_X=False):
        """
        The model forward pass. In [1] and [2], the model estimates the score function. In [3], the model estimates
        either the score function or the target data for the Schrödinger bridge (loss_type='data_prediction').

        [1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, and  Timo Gerkmann
            "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models"
            IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023.

        [2] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann
            "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation"
            ISCA Interspecch, Kos, Greece, Sept. 2024.

        [3] Julius Richter, Danilo de Oliveira, and Timo Gerkmann
            "Investigating Training Objectives for Generative Speech Enhancement"
            https://arxiv.org/abs/2409.10753

        """

        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            F = self.dnn(self._c_in(t) * x_t, self._c_in(t) * y, t)

            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]

            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * x_t + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (F - x_t) / sigmas.pow(2)
                return score
            elif self.loss_type == "precond_denoiser":  # for M3
                D = self._c_skip(t) * x_t + self._c_out(t) * F
                if return_X:
                    # print('c_in:{},c_out:{}c_skip:{}'.format(self._c_in(t), self._c_out(t), self._c_skip(t)))
                    # for N=1, the output is :
                    # c_in:tensor([[[[2.4898]]]], device='cuda:0')
                    # c_out:tensor([[[[0.0969]]]], device='cuda:0')
                    # c_skip:tensor([[[[0.0620]]]], device='cuda:0')
                    return D
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (D - x_t) / sigmas.pow(2)
                return score
            elif self.loss_type == 'data_prediction':
                x_hat = self._c_skip(t) * x_t + self._c_out(t) * F
                return x_hat

        # In [1] and [2], we use the old code:
        else:
            dnn_input = torch.cat([x_t, y], dim=1)
            # print('dnn_input:{}, t:{}, next(target_score_model.dnn.parameters()).device:{}'.format(
            # dnn_input.device, t.device, next(self.dnn.parameters()).device))
            score = -self.dnn(dnn_input, t)
            # print('dnn_input:{}, score:{}, score_device: {}, t:{}'.format(
            # torch.mean(dnn_input), torch.mean(score), score.device,  t))
            return score

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_in type: {}".format(self.c_in))

    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None]
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_out type: {}".format(self.c_out))

    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        if self.c_skip == "1":
            return 1.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_skip type: {}".format(self.c_skip))

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_consistency_sampler(self, consistency_sampler_name, y, N=None, ts=None, snr=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N
        return sampling.get_consistency_sampler(consistency_sampler_name, sde=sde, score_fn=self,
                                                y=y,
                                                eps=self.t_eps,
                                                ts=ts,
                                                snr=snr,
                                                **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(
                        sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
                corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
                **kwargs
                ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)

        # SGMSE sampling with OUVE SDE
        if self.sde.__class__.__name__ == 'OUVESDE':
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N,
                                              corrector_steps=corrector_steps, snr=snr, intermediate=False,
                                              **kwargs)
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
            else:
                raise ValueError(
                    "Invalid sampler type for SGMSE sampling: {}".format(sampler_type))
        # Schrödinger bridge sampling with VE SDE
        elif self.sde.__class__.__name__ == 'SBVESDE':
            sampler = self.get_sb_sampler(
                sde=self.sde, y=Y.cuda(), sampler_type=self.sde.sampler_type)
        else:
            raise ValueError("Invalid SDE type for speech enhancement: {}".format(
                self.sde.__class__.__name__))

        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/self.sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
