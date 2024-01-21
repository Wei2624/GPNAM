import argparse
import os
from os.path import join as pjoin, exists as pexists
import shutil
import torch

import gpnam

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--name",
                        default='debug',
                        help="Name of this run. Used for monitoring and checkpointing.")
    # Load hparams from a model name
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for initializing training.')
    parser.add_argument("--dataset", default='LCD',
                        help="Choose the dataset.",
                        choices=['LCD', 'GMSC', 'CAHousing'])
    parser.add_argument("--optimizer", default='Adam',
                        help="Choose the optimizer.",
                        choices=['Adam', 'SGD', 'CG'])
    parser.add_argument('--fold', type=int, default=0,
                        help='Choose from 0 to 4, as we only support 5-fold CV.')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--quantile_dist", type=str, default='normal',
                        choices=['normal', 'uniform'],
                        help='Which distribution to do qunatile transform')

    parser.add_argument("--early_stopping_rounds", type=int, default=11000)
    parser.add_argument("--max_rounds", type=int, default=-1)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--max_time", type=float, default=3600 * 20)  # At most 20 hours
    parser.add_argument("--report_frequency", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_bs", type=int, default=2048,
                        help='If batch size is None, it automatically finds the right batch size '
                             'that fits into the GPU memory between max_bs and min_bs via binary '
                             'search.')
    parser.add_argument("--min_bs", type=int, default=128)

    temp_args, _ = parser.parse_known_args()
    # Remove stuff if in debug mode
    if temp_args.name.startswith('debug'):
        clean_up(temp_args.name)

    args = parser.parse_args()

    return args

def clean_up(name):
    shutil.rmtree(pjoin('logs', name), ignore_errors=True)
    shutil.rmtree(pjoin('lightning_logs', name), ignore_errors=True)
    if pexists(pjoin('logs', 'hparams', name)):
        os.remove(pjoin('logs', 'hparams', name))


def main():
    args = get_args()

    # Create directory
    os.makedirs(pjoin('logs', args.name), exist_ok=True)

    # Set seed
    if args.seed is not None:
        gpnam.utils.seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset, test_dataset, task = gpnam.data.get_dataset(args.dataset)

    if task == 'classification':
        model = gpnam.model.GPNAMClass(train_dataset.get_input_dim())
    elif task == 'regression':
        model = gpnam.model.GPNAMReg(train_dataset.get_input_dim(), train_dataset.get_kernel_width())
    else:
        raise NotImplementedError()

    trainer = gpnam.trainer.Trainer(model, train_dataset, test_dataset, batch_size=args.batch_size, problem=task, optimizer=args.optimizer, n_epochs=args.n_epochs)

    trainer.train(device)

    trainer.evaluate(device)



if __name__ == '__main__':
    main()