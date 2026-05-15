import os
import argparse
import pytorch_lightning as pl
from sgmse.model import ScoreModel
from sgmse.sdes import SDERegistry
from sgmse.data_module import SpecsDataModule
from sgmse.backbones.shared import BackboneRegistry
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from os.path import join

# Set CUDA architecture list and float32 matmul precision high
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

# Suppress PyTorch AccumulateGrad stream warning (common in PyTorch Lightning + DDP)
if hasattr(torch.autograd.graph, 'set_warn_on_accumulate_grad_stream_mismatch'):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


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

    temp_args, _ = base_parser.parse_known_args()

    # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    sde_class = SDERegistry.get_by_name(temp_args.sde)
    trainer_parser = parser.add_argument_group(
        "Trainer", description="Lightning Trainer")
    trainer_parser.add_argument("--accelerator", type=str, default="gpu",
                                help="Supports passing different accelerator types.")
    trainer_parser.add_argument(
        "--devices", default="auto", help="How many gpus to use.")
    trainer_parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
    trainer_parser.add_argument(
        "--max_epochs", type=int, default=120, help="Number of epochs to train.")

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
    print("=" * 40)
    print("Model (teacher_model):")
    print("=" * 40)
    print(f"{'backbone':20}: {model.backbone}")
    print(f"{'c_in':20}: {model.c_in}")
    print(f"{'c_out':20}: {model.c_out}")
    print(f"{'c_skip':20}: {model.c_skip}")
    print(f"{'loss_type':20}: {model.loss_type}")
    print(f"{'loss_weighting':20}: {model.loss_weighting}")
    print(f"{'l1_weight':20}: {model.l1_weight}")
    print(f"{'t_eps':20}: {model.t_eps}")
    print(f"{'pesq_weight':20}: {model.pesq_weight}")
    print(f"{'network_scaling':20}: {model.network_scaling}")
    print(f"{'sigma_data':20}: {model.sigma_data}")
    print(f"{'num_eval_files':20}: {model.num_eval_files}")
    print(f"{'sr':20}: {model.sr}")
    print(f"{'sde.N':20}: {model.sde.N}")
    print("\n")

    # Set up logger configuration
    if args.nolog:
        logger = None
    else:
        logger = WandbLogger(project="sgmse", log_model=False,
                             save_dir=args.log_dir, name=args.wandb_name)
        logger.experiment.log_code(".")
    os.makedirs(args.log_dir, exist_ok=True)
    # Set up callbacks for logger
    if logger != None:

        callbacks = [
            ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_last=True,
                filename='{epoch}-last'),
            # Save every 10th epoch
            ModelCheckpoint(
                dirpath=join(args.log_dir, str(logger.version)),
                save_top_k=-1,  # Save all checkpoints that meet the condition
                every_n_epochs=10,
                filename='{epoch:03d}'
            )
        ]
        callbacks += [ModelCheckpoint(dirpath=join(args.log_dir, f'{str(logger.version)}-{args.wandb_name}'),
                                      filename='{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval)]
        if args.num_eval_files:
            checkpoint_callback_pesq = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)),
                                                       save_top_k=2, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
            checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)),
                                                         save_top_k=2, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
            callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]
    else:
        callbacks = []

    class CustomProgressBar(TQDMProgressBar):
        def get_metrics(self, trainer, model):
            items = super().get_metrics(trainer, model)
            short_items = {}
            for k, v in items.items():
                if isinstance(v, float):
                    v_str = f"{v:.4f}"
                else:
                    v_str = str(v)
                
                # Shorten the verbose metric keys for the progress bar only
                k = k.replace('train_loss_step', 'Loss_s')
                k = k.replace('train_loss_epoch', 'Loss_e')
                k = k.replace('Consistency_loss_step', 'CT_s')
                k = k.replace('Consistency_loss_epoch', 'CT_e')
                k = k.replace('PESQ_loss_step', 'PESQ_s')
                k = k.replace('PESQ_loss_epoch', 'PESQ_e')
                k = k.replace('SISDR_loss_step', 'SDR_s')
                k = k.replace('SISDR_loss_epoch', 'SDR_e')
                
                short_items[k] = v_str
            return short_items
    
    callbacks.append(CustomProgressBar())

    # Initialize the Trainer and the DataModule
    trainer = pl.Trainer(
        **vars(arg_groups['Trainer']),
        strategy="ddp", logger=logger,
        log_every_n_steps=100, num_sanity_val_steps=0,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_model_summary=True,
        limit_val_batches=20,
        check_val_every_n_epoch=10 # Run validation every 10 epochs
    )

    # Train model
    trainer.fit(model, ckpt_path=args.ckpt)
