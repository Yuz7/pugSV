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

import argparse
from pugsv.utils.config_utils import TrainingConfig
import pugsv.utils.data_utils as data_utils
from pugsv.tokenization import tokenization
from torch.utils.data import DataLoader
import torch.optim as optim
import pysam
import gc

parser = argparse.ArgumentParser(description='Cue model training')
parser.add_argument('--config', help='Training config')
parser.add_argument('--data_config', help='(Optional) Dataset config for streaming', default=None)
args = parser.parse_args()

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
    
    for chrom in train_chroms + valid_chroms:
        if chrom not in task_list_bychrom.keys():
            continue
        
        task_list = task_list_bychrom[chrom]
        for task in task_list:
            
            pass
        pass
    
    pass
