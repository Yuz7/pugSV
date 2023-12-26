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
from pysam import AlignedSegment
import re
import logging

MIN_MAPQ = 10
MIN_SV_SIZE = 30

class aligncate:
    def __init__(self, align: AlignedSegment):
        self.align = align
        self.start = self.align.reference_start
        self.end = self.align.reference_end
        self.name = align.query_name
        self.signs = []
        self.__generate_index()
        pass
    
    def char_data(self, start, end):
        inscount, inslen, delcount, dellen = 0, 0, 0, 0
        for sign in self.signs:
            if sign[0][0] > end:
                break
            elif sign[0][0] < start:
                continue
            elif sign[0][0] >= start:
                if sign[2] == 'I':
                    inscount += 1
                    inslen += sign[0][1] - sign[0][0]
                elif sign[2] == 'D':
                    delcount += 1
                    dellen += sign[1][1] - sign[1][0]
        return inscount, inslen, delcount, dellen
        pass
    
    def __generate_index(self):
        if self.align.cigarstring == None:
            return
        
        if self.align.is_unmapped or self.align.is_secondary or self.align.mapq < MIN_MAPQ:
            return
        
        readPos = self.align.reference_start
        refPos = self.align.reference_start
        read_start = self.align.query_alignment_start
        # # traverse cigar to find gaps(I or D) longer than min_sv_size
        cigar_ops, cigar_lengths = self.__cigar_to_list(self.align.cigarstring)
        # read_seq = self.align.query_sequence
        for i in range(len(cigar_ops)):
            op = cigar_ops[i]
            opLen = cigar_lengths[i]

            if op == "N" or op == "S":
                readPos += opLen

            elif op == "I":
                if opLen >= MIN_SV_SIZE:
                    self.signs.append([[readPos, readPos + opLen], [refPos, refPos], 'I'])
                readPos += opLen

            elif op == "D":
                if opLen >= MIN_SV_SIZE:
                    self.signs.append([[readPos, readPos], [refPos, refPos + opLen], 'D'])
                refPos += opLen

            elif op in ["M", "X", "E", '=']:
                refPos += opLen
                readPos += opLen

            elif op == "H":
                pass
            else:
                pass
        
        pass
    
    def __cigar_to_list(self, cigar):
        opVals = re.findall(r'(\d+)([\w=])', cigar)
        lengths = [int(opVals[i][0]) for i in range(0, len(opVals))]
        ops = [opVals[i][1] for i in range(0, len(opVals))]

        return ops, lengths