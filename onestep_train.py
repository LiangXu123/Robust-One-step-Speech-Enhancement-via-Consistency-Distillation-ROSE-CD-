import random
import numpy as np
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning import Trainer
import copy
import warnings
from pytorch_lightning.callbacks import Callback
import warnings
from sgmse.model import ScoreModel
from sgmse.sdes import SDERegistry
from sgmse.data_module import SpecsDataModule
from sgmse.backbones.shared import BackboneRegistry
import torch
import os
import wandb
import argparse
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join

# Set CUDA architecture list and float32 matmul precision high
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

warnings.filterwarnings('ignore')


def get_argparse_groups(parser):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None)
                      for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == '__main__':
    # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument(
            "--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
        parser_.add_argument(
            "--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
        parser_.add_argument("--nolog", action='store_true',
                             help="Turn off logging.")
        parser_.add_argument("--wandb_name", type=str, default=None,
                             help="Name for wandb logger. If not set, a random name is generated.")
        parser_.add_argument("--ckpt", type=str, default=None,
                             help="Resume training from checkpoint.")
        parser_.add_argument("--log_dir", type=str,
                             default="logs", help="Directory to save logs.")
        parser_.add_argument("--save_ckpt_interval", type=int,
                             default=50000, help="Save checkpoint interval.")
        parser_.add_argument("--distill_N", type=int, default=30,
                             help="The number of timesteps in the distill procsee. 30 by default., will be set to model.sde.N after initlized. ")

        # Add distill-specific parameters
        parser_.add_argument("--distill_solver", type=str, default="pc2",
                             choices=["pc2", "mix", "heun", "euler", "sb", "sb_sde"], help="Solver for distillation.")
        parser_.add_argument("--distill_loss_type", type=str, default="L2", choices=[
                             "L2"], help="Loss type for distillation.")
        parser_.add_argument("--distill_sample_return_xmean",
                             action='store_true', help="Turn on return xmean in distillation.")
        parser_.add_argument(
            "--with_pesq_loss", action='store_true', help="Turn on with_pesq_loss.")
        parser_.add_argument(
            "--with_sisdr_loss", action='store_true', help="Turn on with_sisdr_loss.")
        parser_.add_argument("--weight_schedule", type=str, default="uniform", choices=[
            "snr", "snr+1", "karras", "truncated-snr", "uniform", "other_schedule"],
            help="Weight schedule for distillation. Choose from ['snr', 'snr+1', 'karras', 'truncated-snr', 'uniform', 'other_schedule']."
        )

    temp_args, _ = base_parser.parse_known_args()

    # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    sde_class = SDERegistry.get_by_name(temp_args.sde)
    trainer_parser = parser.add_argument_group(
        "Trainer", description="Lightning Trainer")
    trainer_parser.add_argument("--accelerator", type=str, default="gpu",
                                help="Supports passing different accelerator types.")
    trainer_parser.add_argument(
        "--devices", default="1", help="How many gpus to use.")
    trainer_parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
    trainer_parser.add_argument(
        "--max_epochs", type=int, default=100, help="Number of epochs to train.")

    ScoreModel.add_argparse_args(
        parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
    sde_class.add_argparse_args(
        parser.add_argument_group("SDE", description=sde_class.__name__))
    backbone_cls.add_argparse_args(
        parser.add_argument_group("Backbone", description=backbone_cls.__name__))
    # Add data module args
    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(
        parser.add_argument_group("DataModule", description=data_module_cls.__name__))
    # Parse args and separate into groups
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)
    ckpt_path = args.ckpt
    distill_N = args.distill_N
    # Initialize logger, trainer, model, datamodule
    model = ScoreModel(
        backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
        **{
            **vars(arg_groups['ScoreModel']),
            **vars(arg_groups['SDE']),
            **vars(arg_groups['Backbone']),
            **vars(arg_groups['DataModule'])
        }
    )
    checkpoint = torch.load(
        ckpt_path, map_location="cuda:0")  # Load checkpoint
    model.load_state_dict(
        checkpoint["state_dict"], strict=False)  # Load weights

    with torch.no_grad():
        target_model = ScoreModel(
            backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
            **{
                **vars(arg_groups['ScoreModel']),
                **vars(arg_groups['SDE']),
                **vars(arg_groups['Backbone']),
                **vars(arg_groups['DataModule'])
            }
        )
        checkpoint = torch.load(
            ckpt_path, map_location="cuda:0")  # Load checkpoint
        target_model.load_state_dict(
            checkpoint["state_dict"], strict=False)  # Load weights
        if args.distill_solver == 'sb':
            # use sb teacher
            teacher_args = {
                **vars(arg_groups['ScoreModel']),
                **vars(arg_groups['SDE']),
                **vars(arg_groups['Backbone']),
                **vars(arg_groups['DataModule'])
            }
            teacher_args['loss_type'] = "data_prediction"
            teacher_args['c_in'] = '1'
            teacher_args['c_out'] = '1'
            teacher_args['c_skip'] = '0'
            teacher_model = ScoreModel(
                backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                **teacher_args
            )
        else:
            teacher_model = ScoreModel(
                backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                **{
                    **vars(arg_groups['ScoreModel']),
                    **vars(arg_groups['SDE']),
                    **vars(arg_groups['Backbone']),
                    **vars(arg_groups['DataModule'])
                }
            )
        checkpoint = torch.load(
            ckpt_path, map_location="cuda:0")  # Load checkpoint
        teacher_model.load_state_dict(
            checkpoint["state_dict"], strict=False)  # Load weights

    target_model.dnn.train()
    teacher_model.dnn.eval()
    teacher_model.eval(no_ema=True)
    target_model.to("cuda:0")
    teacher_model.to("cuda:0")
    model.to("cuda:0")
    if args.distill_solver == 'sb':
        teacher_model.c_in = '1'
    else:
        # The teacher model supports edm architecture,so we set c_in,c_out,c_skip  to edm, but leave the model with c_in to 1
        teacher_model.c_in = 'edm'

    for param in teacher_model.parameters():
        param.requires_grad = False
    for param in target_model.parameters():
        param.requires_grad = False
    model.train()
    model.dnn.train()

    # set distill timesteps here
    model.sde.N = distill_N
    target_model.sde.N = distill_N
    teacher_model.sde.N = distill_N

    # Print out all the parameters before calling distill_on
    print(f"ckpt_path: {ckpt_path}")
    print("Distillation Parameters:")
    print(f"Distill Solver: {args.distill_solver}")
    print(f"Distill Loss Type: {args.distill_loss_type}")
    print(f"with_pesq_loss: {args.with_pesq_loss}")
    print(f"with_sisdr_loss: {args.with_sisdr_loss}")
    print(f"Distill Sample Return XMean: {args.distill_sample_return_xmean}")
    print(f"Weight Schedule: {args.weight_schedule}")
    # Define the distillation setup for your model
    model.distill_on(
        distill_solver=args.distill_solver,
        distill_loss_type=args.distill_loss_type,
        with_pesq_loss=args.with_pesq_loss,
        with_sisdr_loss=args.with_sisdr_loss,
        distill_sample_return_xmean=args.distill_sample_return_xmean,
        weight_schedule=args.weight_schedule
    )

    class CustomCallback(Callback):
        def __init__(self, target_score_model, teacher_score_model):
            super().__init__()
            self.target_score_model = target_score_model
            self.teacher_score_model = teacher_score_model

        def on_train_start(self, trainer, pl_module):
            """Attach models to `pl_module` before training starts."""
            pl_module.target_score_model = self.target_score_model
            pl_module.teacher_score_model = self.teacher_score_model

    # Initialize callbacks with the custom callback
    callbacks = [CustomCallback(target_model, teacher_model)]

    # Set up logger configuration
    if args.nolog:
        logger = None
    else:
        logger = WandbLogger(project="sgmse-distill", log_model=False,
                             save_dir=args.log_dir, name=args.wandb_name)
        logger.experiment.log_code(".")

    # Ensure log directory exists
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up checkpoint saving - this will work regardless of logger configuration
    if logger is not None:
        # With logger - save to versioned directory
        checkpoint_dir = join(args.log_dir, str(logger.version))
        wandb_checkpoint_dir = join(
            args.log_dir, f'{str(logger.version)}-{args.wandb_name}')
    else:
        # Without logger - save to default directory
        checkpoint_dir = join(args.log_dir, 'no_logger_run')
        wandb_checkpoint_dir = join(args.log_dir, 'no_logger_run_steps')

    # Ensure checkpoint directories exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(wandb_checkpoint_dir, exist_ok=True)

    # Add essential checkpoint callbacks (these will always be added)
    callbacks.extend([
        # Save last checkpoint
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_last=True,
            filename='{epoch}-last'
        ),
        # Save every 10 epochs - THIS IS THE KEY CHANGE
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=-1,  # Save all checkpoints that meet the condition
            every_n_epochs=10,
            filename='{epoch:03d}',
            verbose=True  # Add verbose to confirm saving
        ),
        # Save based on training steps
        ModelCheckpoint(
            dirpath=wandb_checkpoint_dir,
            filename='{step}',
            save_top_k=-1,
            every_n_train_steps=args.save_ckpt_interval
        )
    ])

    # Add metric-based checkpoints only if num_eval_files is available and > 0
    if hasattr(args, 'num_eval_files') and args.num_eval_files and args.num_eval_files > 0:
        callbacks.extend([
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=2,
                monitor="pesq",
                mode="max",
                filename='{epoch}-{pesq:.2f}'
            ),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=2,
                monitor="si_sdr",
                mode="max",
                filename='{epoch}-{si_sdr:.2f}'
            )
        ])

    # Print checkpoint configuration for verification
    print("=" * 50)
    print("CHECKPOINT CONFIGURATION:")
    print("=" * 50)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Step-based checkpoint directory: {wandb_checkpoint_dir}")
    print(f"Saving every 10 epochs: ENABLED")
    print(f"Saving every {args.save_ckpt_interval} steps: ENABLED")
    print(f"Saving last checkpoint: ENABLED")
    if hasattr(args, 'num_eval_files') and args.num_eval_files and args.num_eval_files > 0:
        print(f"Metric-based checkpoints (PESQ/SI-SDR): ENABLED")
    else:
        print(f"Metric-based checkpoints: DISABLED (num_eval_files not set or 0)")
    print("=" * 50)

    for idx, one_model in enumerate([model, teacher_model, target_model]):
        print("=" * 40)
        print(
            f"Model {idx + 1} ({'model' if idx == 0 else 'teacher_model' if idx == 1 else 'target_model'}):")
        print("=" * 40)
        print(f"{'backbone':20}: {one_model.backbone}")
        print(f"{'c_in':20}: {one_model.c_in}")
        print(f"{'c_out':20}: {one_model.c_out}")
        print(f"{'c_skip':20}: {one_model.c_skip}")
        print(f"{'loss_type':20}: {one_model.loss_type}")
        print(f"{'loss_weighting':20}: {one_model.loss_weighting}")
        print(f"{'l1_weight':20}: {one_model.l1_weight}")
        print(f"{'t_eps':20}: {one_model.t_eps}")
        print(f"{'pesq_weight':20}: {one_model.pesq_weight}")
        print(f"{'network_scaling':20}: {one_model.network_scaling}")
        print(f"{'sigma_data':20}: {one_model.sigma_data}")
        print(f"{'num_eval_files':20}: {one_model.num_eval_files}")
        print(f"{'sr':20}: {one_model.sr}")
        print(f"{'with_pesq_loss':20}: {one_model.with_pesq_loss}")
        print(f"{'sde.N':20}: {one_model.sde.N}")
        print("\n")

    # Initialize the Trainer and the DataModule
    trainer = pl.Trainer(
        **vars(arg_groups['Trainer']),
        strategy="auto",
        # strategy=SingleDeviceStrategy(device=main_device),
        logger=logger,
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_model_summary=True,
        limit_val_batches=200,
        check_val_every_n_epoch=10  # Run validation every 10 epochs
    )

    torch.cuda.empty_cache()
    trainer.fit(model)
