import pugsv.utils.config_utils as config_utils
import pugsv.model as models
from torchsummary import summary
import numpy as np
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
import logging

warnings.filterwarnings('ignore')

INTERVAL_SIZE = 1500000

if __name__ == '__main__':
    data_config = load_config("/Users/yuz/Work/SVs/pugSV/pugsv/config.yaml", config_type=config_utils.CONFIG_TYPE.TRAIN)
        #split chorms to train chorms and valid chorms with 4:1
    bam_path = data_config.bam
    
    #train_chroms, valid_chroms = data_utils.data_split(chroms, 0.8)
    
    #debug
    #train_chroms = ['chr1', 'chr2', 'chr3', 'chr4']
    train_chroms = ['14']
    valid_chroms = ['15']
    train_tokens = defaultdict(list)
    logging.info("******************** init ground truth by vcf ********************")
    ground_truth = pugsvIO.BedRecordContainer(data_config.vcf)
    targets = defaultdict(list)
    tokens = defaultdict(list)
    # valid_tokens = []
    pool = multiprocessing.get_context('fork').Pool(processes = 6)
    for chrom in train_chroms + valid_chroms:
        
        #TODO multi-threading by interval_size
        aln_file = pysam.AlignmentFile(bam_path)
        chrom_len = aln_file.get_reference_length(chrom)
        interval_count = chrom_len // INTERVAL_SIZE
        pos = 0
        logging.info("******************** processing bam into tokens with chrom:{0} chrom_len:{1} interval_count:{2} ********************".format(chrom, chrom_len, interval_count))
        for interval_id in range(interval_count):
            pool.apply_async(collect_tokens,(bam_path, pos, interval_id, chrom, chrom_len, INTERVAL_SIZE), 
                                callback=tokens[chrom].extend, error_callback=collect_err_callback)
            pos += INTERVAL_SIZE + 1
            pass
        
        pass
    pool.close()
    pool.join()
    logging.info("******************** start tokenization ********************")
    for chrom in train_chroms + valid_chroms:
        logging.info("******************** tokenization chrom{0} ********************".format(chrom))
        train_tokens[chrom].extend(tokenization(tokens[chrom]))
    
        for token in train_tokens[chrom]:
            targets[chrom].append(ground_truth.get_sv_type(chrom, token[5], token[6]))
        pass
    
    train.train(targets, train_tokens, train_chroms, valid_chroms)