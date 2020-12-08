#!/usr/bin/env python
# -*- coding:utf-8 -*-
#########################################################################
# File Name: convert_to_bert_keyword.py
# Author: changyumiao
# Mail: changyumiao@tencent.com
# Created Time: Tue 30 Jul 2019 09:30:00 PM CST
# Python Version: 3.6
#########################################################################

"""This script convert keywords in text to keyword mask
Example
---
Input line:
0 \t 中 国 niu## #iub# ###bi 极 了 \t 中国 niubi \t 厉 害 了 我 的 国 \t 厉害 国
Output line:
0 \t 同上 \t 1 1 1 1 1 0 0 \t 同上 \t 1 1 0 0 0 1

"""

import re
import sys


def match_en(s, kw):
    kw_index = []
    for idx,e in enumerate(s):
        e.replace('#', '')
        if e in kw:
            kw_index.append(idx)
    return kw_index


def match_ch(s, kw):
    kw_index = []
    cur_rs = []
    p1 = 0
    p2 = 0
    while(p1+p2 <= len(s)):
        if p2 == len(kw): # match succed
            kw_index += cur_rs
            p2 = 0
            p1 += 1
            cur_rs = []
        if (p1 + p2) < len(s) and s[p1+p2] == kw[p2]: # matching
            cur_rs.append(p1+p2)
            p2 += 1
        else: # match failed
            p2 = 0
            p1 += 1
            cur_rs = []
    return kw_index


def match(s, kws):
    kw_index = set()
    for kw in kws:
        if re.match(r'^[\u4e00-\u9fff]+$', kw):
            kw_index |= set(match_ch(s, kw))
        elif re.match(r'^[a-zA-Z]+$', kw):
            kw_index |= set(match_ch(s, kw))
        else:
            continue
    return kw_index


def main(in_file, out_file, drop_no_kw=None):
    with open(in_file, 'r', encoding='utf8') as fi, \
         open(out_file, 'w', encoding='utf8') as fo:
        for idx, line in enumerate(fi):
            print('>> processing line %d' % idx, end='\r')
            line = line.strip('\n').split('\t')
            if len(line) != 5:
                continue
            label = line[0]
            text_a, kw_a = line[1].split(), line[2].split()
            text_b, kw_b = line[3].split(), line[4].split()
            if drop_no_kw and (not (kw_a and kw_b)): # keep sentence pairs both contain keywords
                continue
            kw_a_index = match(text_a, kw_a)
            kw_b_index = match(text_b, kw_b)
            kw_a_mask = ' '.join(['1' if idx in kw_a_index else '0' 
                                  for idx in range(len(text_a))])
            kw_b_mask = ' '.join(['1' if idx in kw_b_index else '0' 
                                  for idx in range(len(text_b))])
            new_line = '\t'.join([label, line[1], kw_a_mask,
                                         line[3], kw_b_mask])
            print(new_line, file=fo)
            if (idx+1) % 5000 == 0:
                fo.flush()


def test_match_ch(s, kw):
    a = match_ch("ababcdaaabcab", "abc")
    print(a)
    b = match_ch("中 国".split(),"中国")
    print(b)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise ValueError("Invalid arg number!")
