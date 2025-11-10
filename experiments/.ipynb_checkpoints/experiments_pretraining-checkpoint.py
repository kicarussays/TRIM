import os
import torch
import torch.multiprocessing as mp
import argparse
import yaml
from transformers import BertConfig

torch.set_num_threads(32)

parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, 
                    help='cpu or GPU Device Number', default=0)
parser.add_argument('--seed', type=int, 
                    help='Random seed', default=100)
parser.add_argument('--hospital', type=str, 
                    help='mimic or snuh')
parser.add_argument('--model', type=str, 
                    help='behrt, medbert, clmbr')
parser.add_argument('--multi-gpu', action='store_true', 
                    help='Use multi gpu?')

# argument for multi-gpu
parser.add_argument('--num-workers', type=int, default=8, help='')
parser.add_argument("--gpu-devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world-size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()

if args.multi_gpu:
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    os.environ["NCCL_TIMEOUT"] = '1200'
    os.environ["MASTER_PORT"] = '12355'
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size


import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import seedset, pickleload, DotDict
from src.dataset import MLM_Dataset
from src.trainer import EHRTrainer
from src.baseline_models import (
    BehrtForMaskedLM,
    MedbertForMaskedLM)
configpath = os.path.join(abspath, f'configs/for_pretrain/{args.model}.yaml')


abspath = str(Path(__file__).resolve().parent.parent.parent)
spath = os.path.join(abspath, f'usedata/{args.hospital}/')
tpath = os.path.join(abspath, f'usedata/{args.hospital}/tokens/')
rpath = os.path.join(abspath, f'results/{args.hospital}/pretraining/')


def main():
    with open(configpath, 'r') as f: 
        config = DotDict(**yaml.safe_load(f))
        args.bs = config.trainconfig.bs
        args.lr = config.trainconfig.lr
        args.max_epoch = config.trainconfig.max_epoch
            
    seedset(args.seed)
    os.makedirs(os.path.join(rpath, f'logs'), exist_ok=True)
    os.makedirs(os.path.join(rpath, f'saved'), exist_ok=True)

    args.logpth = os.path.join(rpath, f'logs/{args.model}.log')
    args.savepth = os.path.join(rpath, f'saved/{args.model}.tar')
    
    vocab = torch.load(os.path.join(spath, 'vocab.pt'))
    load_dataloader = {}
    for file in ('train', 'valid'):
        token = pickleload(os.path.join(tpath, f'Pretraining_tokens_{file}.pkl'))
        token['age'] = torch.clamp(token['age'], min=0, max=config.bertconfig.max_age-1)
        if args.model != 'behrt-de':
            token = {k: v for k, v in token.items() if k != 'domain'}
        if args.model == 'medbert':
            token['plos_target'] = torch.Tensor(
                pickleload(os.path.join(tpath, f'Pretraining_plos_{file}.pkl')))
        dataset = MLM_Dataset(
            token, 
            vocabulary=vocab, 
            masked_ratio=config.datasetconfig.masked_ratio,
            ignore_special_tokens=True,)
        load_dataloader[file] = dataset

    bertconfig = BertConfig(vocab_size=len(vocab), **config.bertconfig)
    if args.model in ('behrt', 'behrt-de'):
        model = BehrtForMaskedLM(bertconfig)
    elif args.model == 'medbert':
        model = MedbertForMaskedLM(bertconfig)
    else:
        assert "Cannot applicable the model"

    
    Trainer = EHRTrainer(
        model=model,
        train_dataset=load_dataloader['train'],
        valid_dataset=load_dataloader['valid'],
        args=args,
    )

    if args.multi_gpu:
        mp.spawn(Trainer.train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        Trainer.train(args=args)


if __name__ == '__main__':
    main()

