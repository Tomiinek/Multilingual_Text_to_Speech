#encoding=utf-8

import fileinput
import pinyin
import jieba

with open('pinyin_output.txt', 'w') as f: 
    for s in fileinput.input():
        print(' '.join([pinyin.get(x) for x in jieba.cut(s.rstrip())]), file=f)

