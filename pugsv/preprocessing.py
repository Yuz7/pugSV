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
from pugsv.tokenization import char
from pugsv.tokenization import tokenization

MIN_TOKEN_SIZE = 100

def preprocessing(bam_path, chrom, interval_size):
    aln_file = pysam.AlignmentFile(bam_path)
    chrom_len = aln_file.get_reference_length(chrom)
    interval_count = chrom_len // interval_size
    pos = 0
    new_aligns = [aligncate]
    chars = []     #inscount   inslen  delcount    dellen  depth   pos_start    pos_end
    tokens = [token]

    for interval_id in range(interval_count):
        start = pos
        end = pos + interval_size if pos + interval_size > chrom_len else chrom_len
        aligns = aln_file.fetch(chrom, start, end)
        for align in aligns:
            new_aligns.append(aligncate(align))
            pass
        del aligns
        token_iter_pos = pos
        while(True):
            for new_align in new_aligns:
                if new_align.start > token_iter_pos:
                    break
                if new_align.end <= token_iter_pos:
                    continue
                chars.append(char(new_align, MIN_TOKEN_SIZE, interval_id, token_iter_pos, token_iter_pos + MIN_TOKEN_SIZE))
                pass
            tokens.append(token(chars))
            if token_iter_pos + MIN_TOKEN_SIZE >= end:
                break
            token_iter_pos += MIN_TOKEN_SIZE + 1
            pass
        
        if(end == chrom_len):
            break
    
    return tokenization(tokens)