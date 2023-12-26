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
import pugsv.utils.config_utils as config_utils
from pugsv.utils.config_utils import TrainingConfig,load_config
from pugsv.tokenization import tokenization
import pugsv.utils.io as pugsvIO
import pugsv.utils.data_utils as data_utils
import pugsv.model as models
from pugsv.dataset import pugDataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import pysam, torch, time, platform, os, multiprocessing
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

INTERVAL_SIZE = 1500000
BATCH_SIZE = 128

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model and updata weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device) 
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += (exp.shape[0] * exp.shape[1])
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=2)[1]
        result = classification_report(pred_classes.view(-1), label.to(device).view(-1), output_dict=True)
        loss = loss_function(pred.permute(0,2,1), label.to(device).long())
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               result['macro avg']['f1-score'])
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step() 
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), result['macro avg']['f1-score']

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
        sample_num += (exp.shape[0] * exp.shape[1])
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=2)[1]
        result = classification_report(pred_classes.view(-1), labels.to(device).view(-1), output_dict=True)
        loss = loss_function(pred.permute(0,2,1), labels.to(device).long())
        accu_loss += loss
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               result['macro avg']['f1-score'])
    return accu_loss.item() / (step + 1), result['macro avg']['f1-score']

def train(targets, train_tokens):
    np.savetxt('targets.csv', np.array(targets['15']))
    np.savetxt('train_tokens.csv', np.array(train_tokens['15']))
    train_data = []
    train_targets = []
    idx = 0
    while True:
        end = idx + 50 if idx + 50 <= len(train_tokens['15']) else len(train_tokens['15'])
        train_data.append(train_tokens['15'][idx:end])
        train_targets.append(targets['15'][idx:end])
        idx = end
        if end == len(train_tokens['15']):
            break
        pass
    if len(train_data[-1]) < 50:
        train_data[-1].extend([[0] * 7] * (50 - len(train_data[-1])))
        train_targets[-1].extend([0] * (50 - len(train_targets[-1])))
    split_size = math.ceil(len(train_data) * 0.7)
    valid_data = torch.from_numpy(np.array(train_data[split_size:len(train_data)])[:,:,:5]).to(torch.bfloat16)
    train_data = torch.from_numpy(np.array(train_data[0:split_size])[:,:,:5]).to(torch.bfloat16)
    valid_targets = torch.from_numpy(np.array(train_targets[split_size:len(train_targets)]))
    train_targets = torch.from_numpy(np.array(train_targets[0:split_size]))
    
    train_dataset = pugDataset(train_data, train_targets)
    valid_dataset = pugDataset(valid_data, valid_targets)
    
    train_loader =  DataLoader(train_dataset,
                                batch_size=4,
                                shuffle=False,
                                pin_memory=True,drop_last=True)
    valid_loader =  DataLoader(valid_dataset,
                                batch_size=4,
                                shuffle=False,
                                pin_memory=True,drop_last=True)
    
    print("******************** Start trainning ********************")
    print("train_data shape:{0} train_targets shape:{1} train_dataset len:{2}".format(train_data.shape, train_targets.shape, len(train_dataset)))
    
    tb_writer = SummaryWriter()
    device = "cuda:0"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    project_path = '/Users/yuz/Work/SVs/pugSV/project'
    model = models.create_model(4)
    lr = 0.0001
    lrf = 0.01
    epochs = 10
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5) 
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(epochs):
        train_loss, train_f1 = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
        val_loss, val_f1 = evaluate(model=model,
                                     data_loader=valid_loader,
                                     device=device,
                                     epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_f1, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_f1, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if platform.system().lower() == 'windows':
            torch.save(model.state_dict(), project_path+"/model-{}.pth".format(epoch))
        else:
            torch.save(model.state_dict(), "%s"%project_path+"/model-{}.pth".format(epoch))
        pass
    pass