import numpy as np
import os
import sys
from tqdm import tqdm
import time
import datetime
import logging
import traceback
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from torch.optim import AdamW
import torch.distributed as dist

import sys
from pathlib import Path
abspath = str(Path(__file__).resolve().parent.parent)
sys.path.append(abspath)

from src.stat_function import (
    auc_ci, prc_ci, calculate_confidence_interval,
    calculate_sensitivity, calculate_specificity, 
    calculate_precision, calculate_f1_score
)
from src.utils import r3, picklesave


class EHRTrainer:
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        valid_dataset: Dataset = None,
        args: dict = {},
    ):
        self.device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.01,
            eps=1e-8,
        )
        self.args = args

    def train(self, gpu=None, ngpus_per_node=None, args=None):
        if self.args.multi_gpu:
            self.args = args
            self.device = torch.device(f'cuda:{gpu}')
            ngpus_per_node = torch.cuda.device_count()    
            print("Use GPU: {} for training".format(gpu))
        
            self.args.rank = self.args.rank * ngpus_per_node + gpu    
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)
            
            torch.cuda.set_device(gpu)
            self.model.to(self.device)
            self.args.bs = int(self.args.bs / ngpus_per_node)
            self.args.num_workers = int(self.args.num_workers / ngpus_per_node)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[gpu])
        

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.args.logpth)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        first_iter = 0
        if os.path.exists(self.args.savepth):
            print('Saved file exists.')
            checkpoint = torch.load(self.args.savepth, map_location=self.device)
            first_iter = checkpoint["epoch"] + 1
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
        if self.args.multi_gpu:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, 
                                    shuffle=(train_sampler is None), num_workers=self.args.num_workers, 
                                    sampler=train_sampler, pin_memory=True)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
            self.valid_dl = DataLoader(self.valid_dataset, batch_size=self.args.bs, 
                                    shuffle=(valid_sampler is None), num_workers=self.args.num_workers, 
                                    sampler=valid_sampler, pin_memory=True)
        else:
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True)
            self.valid_dl = DataLoader(self.valid_dataset, batch_size=self.args.bs, shuffle=False)

        best_loss = 999999
        patience = 0
        prev_time = time.time()
        for epoch in range(first_iter, self.args.max_epoch):
            self.epoch = epoch
            self.model.train()
            epoch_loss = 0

            for i, batch in enumerate(self.train_dl):
                # Train step
                outputs = self.forward_pass(batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.item()

                # Determine approximate time left
                batches_done = epoch * len(self.train_dl) + i + 1
                batches_left = self.args.max_epoch * len(self.train_dl) - batches_done
                time_left = datetime.timedelta(
                    seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                    % (epoch+1, self.args.max_epoch, i+1, len(self.train_dl), loss.item(), time_left)
                )

            # Validate (returns None if no validation set is provided)
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for _n, batch in enumerate(self.valid_dl):
                    outputs = self.forward_pass(batch)
                    val_loss += outputs.loss.item()
                
            self.val_loss = val_loss / len(self.valid_dl)

            valid_flag = False
            if not self.args.multi_gpu:
                valid_flag = True
            elif self.args.rank == 0:
                valid_flag = True
                    
            if valid_flag:
                # Print epoch info
                self.logger.info('Epoch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f}'.format(
                    epoch+1, self.args.max_epoch, epoch_loss / len(self.train_dl), self.val_loss))

                if self.val_loss < best_loss:
                    self.logger.info('Save best model...')
                    self.save_model()
                    best_loss = self.val_loss
                    patience = 0
                # else:
                #     patience += 1
                #     if patience >= 5: 
                #         self.logger.info('Early stopping activated.')
                #         break


        save_flag = False
        if not self.args.multi_gpu:
            save_flag = True
        else:
            if self.args.rank == 0:
                save_flag = True
        
        if save_flag: self.save_model()


    def forward_pass(self, batch: dict):
        self.to_device(batch)
        model_input = {
            'input_ids': batch['concept'],
            'attention_mask': batch['attention_mask'],
            'segment_ids': batch['segment'] if 'segment' in batch else None,
            'age_ids': batch['age'] if 'age' in batch else None,
            'domain_ids': batch['domain'] if 'domain' in batch else None,
            'target': batch['target'] if 'target' in batch else None,
        }
        if 'plos_target' in batch:
            model_input['plos_target'] = batch['plos_target']
        return self.model(**model_input)
        

    def to_device(self, batch: dict) -> None:
        """Moves a batch to the device in-place"""
        for key, value in batch.items():
            batch[key] = value.to(self.device)
            
    
    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, self.args.savepth)
            
    
    def save_model_ext(self):
        torch.save({
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'{self.args.savepth[:-4]}_epoch{self.epoch}.tar')
            
            
class EHRClassifier(EHRTrainer):
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        valid_dataset: Dataset = None,
        test_dataset: Dataset = None,
        args: dict = {},
    ):
        self.device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
            eps=1e-8,
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.args = args

    def train(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.args.logpth)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        first_iter = 0
        if os.path.exists(self.args.savepth):
            print('Saved file exists.')
            checkpoint = torch.load(self.args.savepth, map_location=self.device)
            first_iter = checkpoint["epoch"] + 1
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # For 10 times oversampling
        label = self.train_dataset.outcomes
        if label.sum() / len(label) < 0.1:
            class_sample_count = torch.Tensor([(len(label)-label.sum())/10, label.sum()])
            weight = 1. / class_sample_count.float()
            samples_weight = torch.tensor([weight[t] for t in label.type(torch.long)])

            sampler = WeightedRandomSampler(
                weights=samples_weight,
                num_samples=len(samples_weight),
                replacement=True  # 복원추출(oversampling)을 위해 replacement=True 설정
    )
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, sampler=sampler)
        else:
            self.train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True)
        self.valid_dl = DataLoader(self.valid_dataset, batch_size=256, shuffle=False)
        self.test_dl = DataLoader(self.test_dataset, batch_size=256, shuffle=False)

        best_score = 0
        patience = 0
        prev_time = time.time()
        try:
            import sys
            for epoch in range(first_iter, self.args.max_epoch):
                stop_flag = False
                self.epoch = epoch
                epoch_loss = 0
                for i, batch in enumerate(self.train_dl):
                    self.model.train()
                    # Train step
                    outputs = self.forward_pass(batch)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()

                    # Determine approximate time left
                    batches_done = epoch * len(self.train_dl) + i + 1
                    batches_left = self.args.max_epoch * len(self.train_dl) - batches_done
                    time_left = datetime.timedelta(
                        seconds=batches_left * (time.time() - prev_time))
                    prev_time = time.time()

                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [Train batch loss: %f] ETA: %s"
                        % (epoch+1, self.args.max_epoch, i+1, len(self.train_dl), loss.item(), time_left)
                    )

                    if (i + 1) % int(len(self.train_dl) / 10) == 0:
                        self.validate(self.valid_dl)
                        
                        # Print epoch info
                        self.logger.info('Epoch [{}/{}], Batch [{}/{}], Train Loss: {:.4f} Val Loss: {:.4f} Val AUROC: {:.4f}'.format(
                            epoch+1, self.args.max_epoch, i+1, len(self.train_dl), epoch_loss / len(self.train_dl), self.val_loss, self.roc_auc))

                        if self.roc_auc > best_score:
                            self.logger.info('Save best model...')
                            self.save_model()
                            best_score = self.roc_auc
                            patience = 0
                        else:
                            patience += 1
                            if patience >= 20: 
                                self.logger.info('Early stopping activated.')
                                stop_flag = True; break
                if stop_flag: break

            self.validate(self.test_dl)
            self.logger.info('Test performance -> Test Loss: {:.4f} Test AUROC: {:.4f}'.format(
                self.val_loss, self.roc_auc))

            roc_put = f'{r3(self.roc_auc)} ({r3(self.rocci[0])}-{r3(self.rocci[1])})'
            prc_put = f'{r3(self.prc_auc)} ({r3(self.prcci[0])}-{r3(self.prcci[1])})'
            sen_put = f'{r3(self.senci[0])} ({r3(self.senci[1])}-{r3(self.senci[2])})'
            spe_put = f'{r3(self.speci[0])} ({r3(self.speci[1])}-{r3(self.speci[2])})'
            pre_put = f'{r3(self.preci[0])} ({r3(self.preci[1])}-{r3(self.preci[2])})'
            f1_put = f'{r3(self.f1ci[0])} ({r3(self.f1ci[1])}-{r3(self.f1ci[2])})'
            self.logger.info(f'AUROC: {roc_put}')
            self.logger.info(f'AUPRC: {prc_put}')
            self.logger.info(f'Sensitivity: {sen_put}')
            self.logger.info(f'Specificity: {spe_put}')
            self.logger.info(f'Precision: {pre_put}')
            self.logger.info(f'F1-score: {f1_put}')

            picklesave((self.val_logits, self.val_labels), f'{self.args.savepth[:-4]}_logits.pkl')
        

        except KeyboardInterrupt:
            import sys
            sys.exit('KeyboardInterrupt')

        except:
            logging.error(traceback.format_exc())


    def validate(self, dl, thres='youden'):
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            val_logits = []
            val_labels = []
            for batch in tqdm(dl):
                outputs = self.forward_pass(batch)
                val_loss += outputs.loss.item()
                val_logits.append(self.softmax(outputs.logits)[:, 1])
                val_labels.append(batch['target'])
            
            self.val_loss = val_loss / len(dl)
            self.val_logits = torch.cat(val_logits).cpu().detach().numpy()
            self.val_labels = torch.cat(val_labels).cpu().detach().numpy()

            fpr, tpr, thresholds = roc_curve(self.val_labels, self.val_logits)
            precision, recall, thresholds_pr = precision_recall_curve(self.val_labels, self.val_logits)

            J = tpr - fpr
            F1s = 2 * (precision * recall) / (precision + recall + 1e-5)
            if thres == 'youden':
                ix = np.argmax(J)
                best_thresh = thresholds[ix] 
            elif thres == 'f1':
                ix = np.argmax(F1s)
                best_thresh = thresholds_pr[ix] 
            else:
                ix = np.where(tpr <= thres)[0][-1] + 1
                best_thresh = thresholds[ix] 
            
            y_prob_pred = (self.val_logits >= best_thresh).astype(bool)

            self.roc_auc = auc(fpr, tpr)
            self.prc_auc = auc(recall, precision)
            self.rocci = auc_ci(self.val_labels, self.val_logits)
            self.prcci = prc_ci(self.val_labels, self.val_logits)
            self.senci = calculate_confidence_interval(calculate_sensitivity, self.val_labels, y_prob_pred)
            self.speci = calculate_confidence_interval(calculate_specificity, self.val_labels, y_prob_pred)
            self.preci = calculate_confidence_interval(calculate_precision, self.val_labels, y_prob_pred)
            self.f1ci = calculate_confidence_interval(calculate_f1_score, self.val_labels, y_prob_pred)
        
            

