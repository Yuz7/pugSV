# MIT License
#
# Copyright (c) 2022 Victoria Popic
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse, gc, sys, math
from pugsv.utils.config_utils import TrainingConfig
from pugsv.tokenization import tokenization
from pugsv.preprocessing import preprocessing
import pugsv.model as models
from pugsv.dataset import pugDataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import pysam, torch, tqdm, time, platform, os
import numpy as np

parser = argparse.ArgumentParser(description='Cue model training')
parser.add_argument('--config', help='Training config')
parser.add_argument('--data_config', help='(Optional) Dataset config for streaming', default=None)
args = parser.parse_args()

INTERVAL_SIZE = 150000
BATCH_SIZE = 8

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model and updata weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss() 
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        loss = loss_function(pred, label.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step() 
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def train(data_config: TrainingConfig, data_loader: DataLoader, epoch, collect_data_metrics=False, classify=False):
    #split chorms to train chorms and valid chorms with 4:1
    bam_path = data_config.bam
    aln_file = pysam.AlignmentFile(bam_path)
    task_list_bychrom = {}
    ref_info = aln_file.get_index_statistics()

    all_possible_chrs = pysam.FastaFile(data_config.genome).references
    tokenization = tokenization(30, data_config)
    
    chroms = []
    for ele in ref_info:
        chrom = ele[0]
        chroms.append(chrom)
        local_ref_len = aln_file.get_reference_length(chrom)

        if chrom not in all_possible_chrs:
            continue

        #TODO window_size have some if else
        window_size = data_config.interval_size

        if local_ref_len < window_size:
            if chrom in task_list_bychrom:
                task_list_bychrom[chrom].append([0, local_ref_len])
            else:
                task_list_bychrom[chrom] = [[0, local_ref_len]]
        else:
            pos = 0
            round_task_num = int(local_ref_len / window_size)
            for j in range(round_task_num):
                if chrom in task_list_bychrom:
                    task_list_bychrom[chrom].append([pos, pos + window_size])
                else:
                    task_list_bychrom[chrom] = [[pos, pos + window_size]]
                pos += window_size

            if pos < local_ref_len:
                if chrom in task_list_bychrom:
                    task_list_bychrom[chrom].append([pos, local_ref_len])
                else:
                    task_list_bychrom[chrom] = [[pos, local_ref_len]]
                    pass
                pass
            pass
        pass
        
    
    # memory opt
    del all_possible_chrs, aln_file
    gc.collect()
    
    #train_chroms, valid_chroms = data_utils.data_split(chroms, 0.8)
    
    #debug
    train_chroms = ['chr1', 'chr2', 'chr3', 'chr4']
    valid_chroms = ['chr15']
    train_tokens = []
    valid_tokens = []
    for chrom in train_chroms:
        if chrom not in task_list_bychrom.keys():
            continue
        
        task_list = task_list_bychrom[chrom]
        for task in task_list:
            train_tokens.append(preprocessing(bam_path, chrom, INTERVAL_SIZE))
            pass
        pass
    
    for chrom in valid_chroms:
        if chrom not in task_list_bychrom.keys():
            continue
        
        task_list = task_list_bychrom[chrom]
        for task in task_list:
            valid_tokens.append(preprocessing(bam_path, chrom, INTERVAL_SIZE))
            pass
        pass
    
    train_data = torch.from_numpy(np.array(train_tokens).astype(np.int64))
    valid_data = torch.from_numpy(np.array(valid_tokens).astype(np.int64))
    
    train_dataset = pugDataset(train_data)
    valid_dataset = pugDataset(valid_data)
    
    train_loader =  DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                pin_memory=True,drop_last=True)
    valid_loader =  DataLoader(valid_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                pin_memory=True,drop_last=True)
    
    tb_writer = SummaryWriter()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    project_path = '../project/'
    model = models.create_model(2, 7)
    lr = 0.0001
    lrf = 0.01
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5) 
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=valid_loader,
                                     device=device,
                                     epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if platform.system().lower() == 'windows':
            torch.save(model.state_dict(), project_path+"/model-{}.pth".format(epoch))
        else:
            torch.save(model.state_dict(), "/%s"%project_path+"/model-{}.pth".format(epoch))
        pass
    pass
