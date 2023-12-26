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

from pugsv.aligncate import aligncate

#tokens = []     inscount   inslen  delcount    dellen  depth   pos_start    pos_end

class token():
    def __init__(self) -> None:
        self.data = [0] * 7
        self.has_data = False
        pass

    def add(self, align: aligncate, token_start: int, token_end: int):
        self.has_data = True
        inscount, inslen, delcount, dellen = align.char_data(token_start, token_end)
        self.data[4] += 1
        self.data[5] = token_start
        self.data[6] = token_end
        ins_multiplier = self.data[0] if self.data[0] > 0 else 1
        self.data[1] = (self.data[1] * ins_multiplier + inslen) // (self.data[0] + inscount) if inslen > 0 and inscount > 0 else self.data[1]
        self.data[0] += inscount
        del_multiplier = self.data[2] if self.data[2] > 0 else 1
        self.data[3] = (self.data[3] * del_multiplier + dellen) // (self.data[2] + delcount) if dellen > 0 and delcount > 0 else self.data[3]
        self.data[2] += delcount
        pass

def tokenization(tokens):
    new_tokens = []
    lastflag = -1
    flag = -1
    merge_tokens = [0] * 7
    for i in range(len(tokens)):
        if tokens[i].data[1] >= 30:
            flag = 0
            pass
        elif tokens[i].data[3] >= 100:
            flag = 1
            pass
        else:
            flag = 2
            pass
        
        if lastflag == -1:
            merge_tokens[5] = tokens[i].data[5]
            merge_tokens[6] = tokens[i].data[6]
        
        dist = merge_tokens[6] - merge_tokens[5] if merge_tokens[6] - merge_tokens[5] > 100 else 100
        if lastflag == -1 or flag == lastflag:
            lastflag = flag
            merge_tokens[6] = tokens[i].data[6]
            merge_tokens[0] += tokens[i].data[0]
            merge_tokens[1] += tokens[i].data[1]
            merge_tokens[2] += tokens[i].data[2]
            merge_tokens[3] += tokens[i].data[3]
            merge_tokens[4] += tokens[i].data[4]
        else:
            lastflag = flag
            merge_tokens[0:5] = [item // (dist // 100) for item in merge_tokens[0:5]]
            new_tokens.append(merge_tokens.copy())
            merge_tokens = [0] * 7
            merge_tokens[5] = tokens[i].data[5]
            merge_tokens[6] = tokens[i].data[6]
            merge_tokens[0] = tokens[i].data[0]
            merge_tokens[1] = tokens[i].data[1]
            merge_tokens[2] = tokens[i].data[2]
            merge_tokens[3] = tokens[i].data[3]
            merge_tokens[4] = tokens[i].data[4]
        
        if i == len(tokens) - 1:
            merge_tokens[0:5] = [item // (dist // 100) for item in merge_tokens[0:5]]
            new_tokens.append(merge_tokens.copy())
        pass
    return new_tokens