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

class char():
    def __init__(self, align: aligncate, min_token_size: int, interval_id: int, token_start: int, token_end: int) -> None:
        self.min_token_size = min_token_size
        self.interval_id = interval_id
        self.align_name = align.name
        self.data = [0] * 7
        self.__convert_to_char(align)
        pass
    
    def __convert_to_token(self, align: aligncate, token_start: int, token_end: int):
        if align.end < token_end:
            token_end = align.end
        self.data[5] = token_start
        self.data[6] = token_end
        self.data[0], self.data[1], self.data[2], self.data[3] = align.char_data(token_start, token_end)
        pass

class token():
    def __init__(self, chars: [char]) -> None:
        self.chars = [char]
        self.data = [0] * 7
        self.__convert_to_token([char])
        pass

    def __convert_to_token(self, chars: [char]):
        for char_loc in range(len(chars)):
            if char_loc == 0:
                self.data[5] = chars[char_loc][5]
            if char_loc == len(chars) - 1:
                self.data[6] = chars[char_loc][6]
            for i in range(4):
                self.data[i] += chars[char_loc][i]
        pass

def __merge_tokens(tokens: [token]):
    merge_tokens = [0] * 7
    for token_loc in range(len(tokens)):
        if token_loc == 0:
            merge_tokens[5] = tokens[token_loc][5]
        if token_loc == len(tokens) - 1:
            merge_tokens[6] = tokens[token_loc][6]
        for i in range(4):
            merge_tokens[i] += tokens[token_loc][i]
    pass

def tokenization(tokens: [token]):
    new_tokens = []
    current_tokens = []
    for token in tokens:
        if token[0] > 0:
            current_tokens.append(token)
            pass
        elif token[3] >= 100:
            current_tokens.append(token)
            pass
        else:
            if current_tokens.count == 0:
                current_tokens.append(token)
            new_tokens.append(__merge_tokens(current_tokens))
            current_tokens = []
    return new_tokens
