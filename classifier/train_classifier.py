import sys
sys.path.insert(1, '/home/zpengac/pose/EFARS/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from glob import glob
import tqdm
import random
from datetime import datetime
import time
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from models.torchvision_models import ResNet18, ResNet50, MobileNetV3Small
from models.gcn import GCNClassifier

from data.human36m import Human36M2DPoseDataset, Human36MMetadata
from utils.misc import AverageMeter, seed_everything
from utils.transform import do_pos2d_train_transforms, do_pos2d_val_transforms
from utils.graph import adj_mx_from_edges

from utils.parser import args

seed_everything(args.seed)

root_path = '/scratch/PI/cqf/datasets/h36m'
img_path = root_path + '/img'
pos2d_path = root_path + '/pos2d'

img_fns = glob(img_path+'/*.jpg')
split = int(0.8*len(img_fns))
random.shuffle(img_fns)
train_fns = img_fns[:50000]
val_fns = img_fns[50000:60000]

# TODO: transforms

train_dataset = Human36M2DPoseDataset(train_fns, pos2d_path, transforms=do_pos2d_train_transforms, out_size=(368, 368), mode='C')
val_dataset = Human36M2DPoseDataset(val_fns, pos2d_path, transforms=do_pos2d_val_transforms, out_size=(368, 368), mode='C')

class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'/home/zpengac/pose/EFARS/classifier/checkpoints/{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.criterion = nn.CrossEntropyLoss().cuda()#ContrastiveLoss()
        #self.metric = torch.dist#nn.CosineSimilarity()
        self.log(f'Fitter prepared. Device is {self.device}')
        
        # self.iters_to_accumulate = 4 # gradient accumulation

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, accuracy = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, ce_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, accuracy: {accuracy.avg:.8f}, time: {(time.time() - t):.5f}')
 
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, accuracy = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, ce_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, accuracy: {accuracy.avg:.8f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        ce_loss = AverageMeter()
        accuracy = AverageMeter()
        t = time.time()
        for step, (imgs, skeletons, labels) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'ce_loss: {ce_loss.avg:.8f}, ' + \
                        f'accuracy: {accuracy.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            with torch.no_grad():
                imgs = imgs.cuda().float()
                skeletons = skeletons.cuda().float()
                labels = labels.cuda()
                batch_size = imgs.shape[0]
                
                with torch.cuda.amp.autocast():
                    preds = self.model(imgs)
                    loss = self.criterion(preds,labels)

            ce_loss.update(loss.detach().item(), batch_size)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            accuracy.update(acc.detach().item(), batch_size)
            #self.scaler.scale(loss).backward()

        return ce_loss, accuracy

    def train_one_epoch(self, train_loader):
        self.model.train()
        ce_loss = AverageMeter()
        accuracy = AverageMeter()
        t = time.time()
        for step, (imgs, skeletons, labels) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'ce_loss: {ce_loss.avg:.8f}, ' + \
                        f'accuracy: {accuracy.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            imgs = imgs.cuda().float()
            skeletons = skeletons.cuda().float()
            labels = labels.cuda()
            batch_size = imgs.shape[0]
            
            with torch.cuda.amp.autocast():
                preds = self.model(imgs)
                loss = self.criterion(preds,labels)

            ce_loss.update(loss.detach().item(), batch_size)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            accuracy.update(acc.detach().item(), batch_size)

            self.scaler.scale(loss).backward()
            # loss = loss / self.iters_to_accumulate # gradient accumulation
            
            #ce_loss.update(loss.detach().item(), batch_size)
            
            #self.optimizer.step()
            self.scaler.step(self.optimizer) # native fp16
            
            if self.config.step_scheduler:
                self.scheduler.step()
            
            self.scaler.update() #native fp16
                
                
#             if (step+1) % self.iters_to_accumulate == 0: # gradient accumulation

#                 self.optimizer.step()
#                 self.optimizer.zero_grad()

#                 if self.config.step_scheduler:
#                     self.scheduler.step()

        return ce_loss, accuracy

    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
            #'amp': amp.state_dict() # apex
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
            
            
class TrainGlobalConfig:
    num_workers = args.num_workers
    batch_size = args.batch_size * torch.cuda.device_count()
    n_epochs = args.n_epochs

    folder = args.output_path
    lr = args.max_lr
    

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = True  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    scheduler_params = dict(
        max_lr=args.max_lr,
        #total_steps = len(train_dataset) // 4 * n_epochs, # gradient accumulation
        epochs=n_epochs,
        steps_per_epoch=int(len(train_dataset) / batch_size),
        pct_start=args.pct_start,
        anneal_strategy=args.anneal_strategy, 
        final_div_factor=args.final_div_factor
    )
    
net = ResNet50(num_classes=Human36MMetadata.num_classes, pretrained=True).cuda()
#net = MobileNetV3Small(num_classes=Human36MMetadata.num_classes, pretrained=True).cuda()
#net = GCNClassifier(adj=adj_mx_from_edges(Human36MMetadata.num_joints, Human36MMetadata.skeleton_edges, sparse=False), hid_dim=128).cuda()

def run_training():
    device = torch.device('cuda')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=SequentialSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=False,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
#     fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')
    fitter.fit(train_loader, val_loader)
    
    
run_training()