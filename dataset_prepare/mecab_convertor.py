#encoding=utf-8

import fileinput
import romkan
import MeCab

wakati = MeCab.Tagger("-Owakati")
yomi = MeCab.Tagger("-Oyomi")

for s in fileinput.input():
    print(romkan.to_roma(yomi.parse(wakati.parse(s))))