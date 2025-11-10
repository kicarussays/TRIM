import os
import sys
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, 
                    help='cpu or GPU Device Number', default=0)
parser.add_argument('--seed', type=int, 
                    help='Random seed', default=100)
parser.add_argument('--hospital', type=str, 
                    help='mimic or ehrshot', default='snuh')
parser.add_argument('--model', type=str, 
                    help='behrt-de, onlyecg', default='behrt-de')
parser.add_argument('--outcome', type=str, 
                    help='Outcomes', default='CVD')
parser.add_argument('--rep-type', type=str,
                    help='Select representation type', default='description')
parser.add_argument('--ehr', action='store_true', 
                    help='Use ECG statements')
parser.add_argument('--ecg', action='store_true', 
                    help='Use ECG statements')
parser.add_argument('--signal', action='store_true', 
                    help='Use ECG signals')
parser.add_argument('--extract-attention-score', action='store_true',
                    help='use adapter')
parser.add_argument('--use-adapter', action='store_true',
                    help='use adapter')
args = parser.parse_args()

abspath = str(Path(__file__).resolve().parent.parent.parent)
spath = os.path.join(abspath, f'usedata/{args.hospital}/')
smpath = os.path.join(abspath, f'usedata/mimic/')
tpath = os.path.join(abspath, f'usedata/{args.hospital}/tokens/Finetuning')
rpath = os.path.join(abspath, f'results/{args.hospital}/ecgfinetuning/')
dpath = os.path.join(abspath, f'usedata/descriptions/')
ppath = os.path.join(abspath, f'results/snuh/pretraining/saved/')
rep_path = os.path.join(abspath, f'usedata/representation')
lpath = os.path.join(abspath, f'results/ecglogits/')
os.makedirs(lpath, exist_ok=True)
os.makedirs(lpath.replace('logits', 'atts'), exist_ok=True)

param_option = f'rep_type={args.rep_type}_ehr={args.ehr}_ecg={args.ecg}_signal={args.signal}'
logit_path = os.path.join(
    lpath, 
    f'{args.model}+{param_option}_{args.outcome}_{args.hospital}')

import numpy as np
import pandas as pd
import torch
import yaml
import copy
from transformers import BertConfig
import ray
import modin.pandas as mpd
import torch.multiprocessing as mp
torch.set_num_threads(8)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)
from src.utils import seedset, pickleload, DotDict
from src.dataset import BinaryOutcomeDataset
from src.trainer import EHRClassifier
from src.baseline_models import EHRECGForSequenceClassification
configpath = os.path.join(abspath, f'configs/{args.model}.yaml')


def main():
    with open(configpath, 'r') as f: 
        config = yaml.safe_load(f)
        args.bs = int(config['trainconfig']['bs'] / 2)
        args.lr = config['trainconfig']['lr']
        args.max_epoch = config['trainconfig']['max_epoch']
        config['bertconfig']['rep_type'] = args.rep_type
        config['bertconfig']['ehr'] = args.ehr
        config['bertconfig']['signal'] = args.signal
        config = DotDict(**config)
        
    device = f'cuda:{args.device}'
    seedset(args.seed)
    os.makedirs(os.path.join(rpath, f'logs'), exist_ok=True)
    os.makedirs(os.path.join(rpath, f'saved'), exist_ok=True)

    pretrained_param = f'all+rep_type={args.rep_type}'
    args.logpth = os.path.join(rpath, f'logs/{args.model}_{param_option}_{args.outcome}_{args.hospital}.log')
    args.savepth = os.path.join(rpath, f'saved/{args.model}_{param_option}_{args.outcome}_{args.hospital}')

    if args.hospital == 'snuh' and args.rep_type == 'none':
        vocab = torch.load(os.path.join(smpath, 'vocab_ecg.pt'))
    else:
        vocab = torch.load(os.path.join(spath, 'vocab_ecg.pt'))
    bertconfig = BertConfig(
        vocab_size=len(vocab), 
        problem_type='multi_label_classification' if args.outcome == 'CVD' else 'single_label_classification',
        use_adapter=args.use_adapter,
        **config.bertconfig)

    # Loading representations for nn.Embedding layers
    selected_representation = None
    if args.rep_type != 'none':
        selected_rep_path = os.path.join(rep_path, f'concept_representation_{args.rep_type}_{args.hospital}ecg.npy')
        selected_representation = np.load(selected_rep_path)

    # Loading training and test data
    ecgoption = 'ecg' if args.ecg else 'noecg'
    token, label = pickleload(os.path.join(tpath, f'{args.outcome}_{args.hospital}_{ecgoption}.pkl'))

    token['age'] = torch.clamp(token['age'], min=0, max=config.bertconfig.max_age-1)
    if args.outcome in ('CVD',):
        label = torch.Tensor(label).type(torch.FloatTensor)
        bertconfig.num_labels = len(label[0])
    else:
        label = torch.Tensor(label).type(torch.LongTensor)
    if args.model != 'behrt-de':
        token = {k: v for k, v in token.items() if k != 'domain'}

    if args.signal:
        signal = np.load(os.path.join(spath, 'all_signals.npy'))
    else:
        signal = None

    dataset = BinaryOutcomeDataset(
        features=token, 
        signal=signal,
        outcomes=label,
        vocabulary=vocab, 
        ignore_special_tokens=True,
    )
    
    # Loading model
    model = EHRECGForSequenceClassification(bertconfig)
    load_pretrained = torch.load(os.path.join(ppath, f'{args.model}_{pretrained_param}.tar'),
                                map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in load_pretrained['model'].items():
        name = k.replace("module.", "")
        if 'concept_embeddings' not in name:
            new_state_dict[name] = v
        if 'domain_embeddings' in name:
            new_state_dict[name] = torch.cat([v, torch.randn(1, 768).to(device)])
    _load = model.load_state_dict(new_state_dict, strict=False)
    print(_load)

    # Loading representations to embedding layers
    if args.rep_type != 'none':
        model.bert.embeddings.concept_embeddings.weight.data = torch.Tensor(selected_representation)
        for param in model.bert.embeddings.parameters(): 
            param.requires_grad_(False)
        
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"{name}: requires_grad = {param.requires_grad}")

    model = model.to(device)
    # Training classifier 
    Trainer = EHRClassifier(
        dataset=dataset,
        ex_dataset=None,
        logit_path=logit_path,
        vocab=vocab,
        ex_vocab=None,
        selected_representation=None if args.rep_type == 'none' else selected_representation,
        ex_selected_representation=None,
        args=args,
        bertconfig=bertconfig
    )

    if not args.extract_attention_score:
        Trainer.process(model, args)
    else:
        Trainer.extract_attention_score(model, args)
    

if __name__ == '__main__':
    main()


