import pugsv.utils.config_utils as config_utils
import pugsv.model as models
from torchsummary import summary
import numpy as np
import argparse, gc, sys, math
from pugsv.utils.config_utils import TrainingConfig
from pugsv.tokenization import tokenization
from pugsv.preprocessing import collect_tokens,collect_err_callback
import pugsv.utils.io as pugsvIO
import pugsv.utils.data_utils as data_utils
import pugsv.model as models
from pugsv.dataset import pugDataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from pugsv.utils.config_utils import load_config
import torch.optim as optim
import pysam, torch, tqdm, time, platform, os, multiprocessing
import pugsv.train as train
import warnings

warnings.filterwarnings('ignore')

INTERVAL_SIZE = 6000000

if __name__ == '__main__':
    data_config = load_config("/Users/yuz/Work/SVs/pugSV/pugsv/config.yaml", config_type=config_utils.CONFIG_TYPE.TRAIN)
        #split chorms to train chorms and valid chorms with 4:1
    bam_path = data_config.bam
    
    #train_chroms, valid_chroms = data_utils.data_split(chroms, 0.8)
    
    #debug
    #train_chroms = ['chr1', 'chr2', 'chr3', 'chr4']
    train_chroms = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
    valid_chroms = ['16']
    train_tokens = defaultdict(list)
    print("******************** init ground truth by vcf ********************")
    ground_truth = pugsvIO.BedRecordContainer(data_config.vcf)
    targets = defaultdict(list)
    tokens = defaultdict(list)
    # valid_tokens = []
    pool = multiprocessing.Pool(processes = 5)
    for chrom in train_chroms + valid_chroms:
        
        #TODO multi-threading by interval_size
        aln_file = pysam.AlignmentFile(bam_path)
        chrom_len = aln_file.get_reference_length(chrom)
        interval_count = chrom_len // INTERVAL_SIZE
        pos = 0
        print("******************** processing bam into tokens with chrom:{0} chrom_len:{1} interval_count:{2} ********************".format(chrom, chrom_len, interval_count))
        for interval_id in range(interval_count):
            pool.apply_async(collect_tokens,(bam_path, pos, interval_id, chrom, chrom_len, INTERVAL_SIZE), 
                                callback=tokens[chrom].extend, error_callback=collect_err_callback)
            pos += INTERVAL_SIZE + 1
            pass
        
        pass
    pool.close()
    pool.join()
    for chrom in train_chroms + valid_chroms:
        train_tokens[chrom].extend(tokenization(tokens[chrom]))
    
        for token in train_tokens[chrom]:
            targets[chrom].append(ground_truth.get_sv_type(chrom, token[5], token[6]))
        pass
    
    train.train(targets, train_tokens, train_chroms, valid_chroms)
# data = torch.from_numpy(np.random.randint(1, 10, size=(50, 50, 6))).to(torch.bfloat16)
# target = torch.from_numpy(np.random.randint(0, 3, size = (50, 50))).to(torch.bfloat16)
# valid_data = torch.from_numpy(np.random.randint(1, 10, size=(50, 50, 6))).to(torch.bfloat16)
# valid_target = torch.from_numpy(np.random.randint(0, 3, size = (50, 50))).to(torch.bfloat16)

# train_dataset = pugDataset(data, target)
# valid_dataset = pugDataset(valid_data, valid_target)

# train_loader =  DataLoader(train_dataset,
#                             batch_size=5,
#                             pin_memory=True,drop_last=True)
# valid_loader =  DataLoader(valid_dataset,
#                             batch_size=5,
#                             pin_memory=True,drop_last=True)

# model = models.create_model(3)
# # summary(model, input_size=[(1, 6)], batch_size=1, device="cpu")

# tb_writer = SummaryWriter()
# device = "cuda:0"
# device = torch.device(device if torch.cuda.is_available() else "cpu")
# project_path = '/Users/yuz/Work/SVs/pugSV/project'
# model = models.create_model(3)
# lr = 0.0001
# lrf = 0.01
# epochs = 10
# pg = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5) 
# lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
# for epoch in range(epochs):
#     train_loss, train_acc = train.train_one_epoch(model=model,
#                                                 optimizer=optimizer,
#                                                 data_loader=train_loader,
#                                                 device=device,
#                                                 epoch=epoch)
#     scheduler.step()
#     val_loss, val_acc = train.evaluate(model=model,
#                                         data_loader=valid_loader,
#                                         device=device,
#                                         epoch=epoch)
#     tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
#     tb_writer.add_scalar(tags[0], train_loss, epoch)
#     tb_writer.add_scalar(tags[1], train_acc, epoch)
#     tb_writer.add_scalar(tags[2], val_loss, epoch)
#     tb_writer.add_scalar(tags[3], val_acc, epoch)
#     tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
#     if platform.system().lower() == 'windows':
#         torch.save(model.state_dict(), project_path+"/model-{}.pth".format(epoch))
#     else:
#         torch.save(model.state_dict(), "%s"%project_path+"/model-{}.pth".format(epoch))
#     pass
# pass