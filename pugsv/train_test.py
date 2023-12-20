import pugsv.utils.config_utils as config_utils
import pugsv.model as models
from torchsummary import summary
import numpy as np
import argparse, gc, sys, math
from pugsv.utils.config_utils import TrainingConfig
from pugsv.tokenization import tokenization
from pugsv.preprocessing import preprocessing
import pugsv.utils.io as pugsvIO
import pugsv.utils.data_utils as data_utils
import pugsv.model as models
from pugsv.dataset import pugDataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import pysam, torch, tqdm, time, platform, os
import pugsv.train as train

data = torch.from_numpy(np.random.randint(1, 10, size=(50, 50, 6))).to(torch.bfloat16)
target = torch.from_numpy(np.random.randint(0, 3, size = (50, 50))).to(torch.bfloat16)
valid_data = torch.from_numpy(np.random.randint(1, 10, size=(50, 50, 6))).to(torch.bfloat16)
valid_target = torch.from_numpy(np.random.randint(0, 3, size = (50, 50))).to(torch.bfloat16)

train_dataset = pugDataset(data, target)
valid_dataset = pugDataset(valid_data, valid_target)

train_loader =  DataLoader(train_dataset,
                            batch_size=5,
                            pin_memory=True,drop_last=True)
valid_loader =  DataLoader(valid_dataset,
                            batch_size=5,
                            pin_memory=True,drop_last=True)

model = models.create_model(3)
# summary(model, input_size=[(1, 6)], batch_size=1, device="cpu")

tb_writer = SummaryWriter()
device = "cuda:0"
device = torch.device(device if torch.cuda.is_available() else "cpu")
project_path = '/Users/yuz/Work/SVs/pugSV/project'
model = models.create_model(3)
lr = 0.0001
lrf = 0.01
epochs = 10
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5) 
lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
for epoch in range(epochs):
    train_loss, train_acc = train.train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
    scheduler.step()
    val_loss, val_acc = train.evaluate(model=model,
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
        torch.save(model.state_dict(), "%s"%project_path+"/model-{}.pth".format(epoch))
    pass
pass