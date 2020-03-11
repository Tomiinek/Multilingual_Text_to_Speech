#encoding=utf-8

import fileinput
import pinyin
import jieba

for s in fileinput.input():
    print(' '.join([pinyin.get(x) for x in jieba.cut(s.rstrip())]))