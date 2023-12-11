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

import yaml
from enum import Enum
import torch
from pathlib import Path
import logging
import sys
import os
import math

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

CONFIG_TYPE = Enum("CONFIG_TYPE", 'TRAIN TEST DATA')

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.experiment_dir = str(Path(config_file).parent.resolve())
        self.devices = []
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
            for i in range(self.n_jobs_per_gpu * len(self.gpu_ids)):
                self.devices.append(torch.device("cuda:%d" % int(i / self.n_jobs_per_gpu)))
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            for _ in range(self.n_cpus):
                self.devices.append(torch.device("cpu"))
            self.device = torch.device("cpu")
        self.log_dir = self.experiment_dir + "/logs/"
        self.report_dir = self.experiment_dir + "/reports/"

        # setup the experiment directory structure
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)

        # logging
        self.log_file = self.log_dir + 'main.log'
        # noinspection PyArgumentList
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.getLevelName(self.logging_level),
                            handlers=[logging.FileHandler(self.log_file, mode='w'), logging.StreamHandler(sys.stdout)])

        logging.info(self)

    def set_defaults(self):
        default_values = {
            'gpu_ids': [],
            'n_cpus': 1,
            'batch_size': 16,
            'logging_level': "INFO",
            'report_interval': 50,
            'n_jobs_per_gpu': 1,
            'signal_set': "SHORT",
            'class_set': "BASIC5ZYG",
            'num_keypoints': 1,
            'model_architecture': "HG",
            'image_dim': 256,
            'sigma': 10,
            'stride': 4,
            'heatmap_peak_threshold': 0.4
        }
        for k, v, in default_values.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)

    def __str__(self):
        s = "==== Config ====\n\t"
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s
    
class TrainingConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        self.set_defaults()
        super().__init__(config_file)
        self.epoch_dirs = []
        for epoch in range(self.num_epochs):
            output_dir = "%s/epoch%d/" % (self.report_dir, epoch)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.epoch_dirs.append(output_dir)

    def __str__(self):
        s = " ===== Training config =====\n\t"
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s

    def set_defaults(self):
        super().set_defaults()
        default_values = {
            'model_checkpoint_interval': 10000,
            'plot_confidence_maps': False,
            'validation_ratio': 0.1,
            'signal_set_origin': "SHORT",
            'learning_rate': 0.0001,
            'learning_rate_decay_interval': 5,
            'learning_rate_decay_factor': 1
        }
        for k, v, in default_values.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)

def load_config(fname, config_type=CONFIG_TYPE.TRAIN):
    """
    Load a YAML configuration file
    """
    with open(fname) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if config_type == CONFIG_TYPE.TRAIN:
        return TrainingConfig(fname, **config)
    # elif config_type == CONFIG_TYPE.TEST:
        # return TestConfig(fname, **config)
    # else:
        # return DatasetConfig(fname, **config)
        