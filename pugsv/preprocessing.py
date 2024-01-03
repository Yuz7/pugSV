# MIT License
#
# Copyright (c) 2022 Yuz7
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

import pysam
from pugsv.aligncate import aligncate
from pugsv.tokenization import token
from pugsv.tokenization import tokenization
from intervaltree import IntervalTree
import time, copy
import logging

MIN_TOKEN_SIZE = 100

def collect_err_callback(err):
    logging.info("error happen:{0}".format(str(err)))

def collect_tokens(bam_path, pos, interval_id, chrom, chrom_len, interval_size):
    aln_file = pysam.AlignmentFile(bam_path)
    start = pos
    end = pos + interval_size if pos + interval_size < chrom_len else chrom_len
    collect_new_aligns = IntervalTree()
    collect_tokens = []
    logging.info("******************** processing interval id:{0} start pos:{1} end pos:{2} ********************".format(interval_id, start, end))
    start_time = time.time()
    aligns = aln_file.fetch(chrom, start, end)
    for align in aligns:
        collect_new_aligns.addi(align.reference_start, align.reference_end, aligncate(align))
        pass
    del aligns
    token_iter_pos = pos
    if len(collect_new_aligns) == 0:
        return collect_tokens
    logging.info("******************** processing interval id:{0} add tokens with collect_new_aligns len:{1}********************".format(interval_id, len(collect_new_aligns)))
    while(True):
        token_temp = token()
        overlap_aligns = collect_new_aligns.overlap(token_iter_pos, token_iter_pos + MIN_TOKEN_SIZE)
        for overlap_align in overlap_aligns:
            token_temp.add(overlap_align.data, token_iter_pos, token_iter_pos + MIN_TOKEN_SIZE)
            pass
        if len(overlap_aligns) > 0:
            collect_tokens.append(copy.deepcopy(token_temp))
        if token_iter_pos + MIN_TOKEN_SIZE >= end:
            break
        token_iter_pos += MIN_TOKEN_SIZE + 1
        pass
    logging.info("******************** processing interval id {0} done, collect_tokens len:{1} and using time:{2} ********************".format(interval_id, len(collect_tokens), (time.time() - start_time) * 1000))
    return collect_tokens