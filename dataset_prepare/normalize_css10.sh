#!/bin/sh

normalize()
{
    FILE=$1

    echo "Cleaning ... $1"

    sed -e 's/\(.*\)|[^|]\+$/\1/' "$FILE" > tmp
    sed -e 's/\(.*\)|[^|]\+$/\1/' tmp > tmp2
    sed -i -e 's/.*|\([^|]\+\)$/\1/' tmp

    # substitute some less frequent characters

    sed -i -e 's/[―—－–]/-/g' tmp
    sed -i -e 's/œ/oe/g' tmp
    sed -i -e 's/æ/ae/g' tmp
    sed -i -e 's/々//g' tmp
    sed -i -e 's/å/a/g' tmp
    sed -i -e 's/ǚ/u/g' tmp
    sed -i -e 's/ǜ/u/g' tmp
    sed -i -e 's/ë/e/g' tmp
    sed -i -e 's/[îïΐ]/í/g' tmp
    sed -i -e 's/ϋ/υ/g' tmp
    sed -i -e 's/ϊ/ι/g' tmp
    sed -i -e 's/！/!/g' tmp
    sed -i -e 's/：/:/g' tmp
    sed -i -e 's/；/;/g' tmp
    sed -i -e 's/？/?/g' tmp
    sed -i -e 's/·/./g' tmp
    sed -i -e "s/’/'/g" tmp
    
    # remove spaces before dots, exclamation marks and etc
    sed -i -e 's/\s\+\([、。，?!,\.:;]\+\)/\1/g' tmp

    # multiple sentence ends change to a single
    sed -i -e 's/:\(\s*[、。，?!,\.:;]\+\)\+/:/g' tmp
    sed -i -e 's/\([?!;\.,]\)[?!;\.,]\+/\1/g' tmp

    # comma, dash in sentence
    sed -i -e 's/,\s\+-/,-/g' tmp

    # some dashes
    sed -i -e 's/\(\s\+\)\(-\+\s*\)\(-\+\s*\)\+/\1/' tmp
    sed -i -e 's/\(\s\+\)\(-\+\s*\)\(-\+\s*\)\+/\1/' tmp    
    sed -i -e 's/^\([^\-]*\)-[ \.?!]\+\([^\-]*\)$/\1\2/' tmp
    sed -i -e 's/^\([^\-]*\)[ \.?!]\+-\([^\-]*\)$/\1\2/' tmp
    sed -i -e 's/^\s*\([、。，?!,\.:;\-]\+\s*\)\+//' tmp
    
    # remove redundant minus
    sed -i -e 's/\([¿?!¡\.:;]\s*\)[\-]\+\s*/\1/g' tmp

    # lines with interpunction only
    sed -i -e "d/^\(\s*[、。，(),\.:;¿?¡!\-]\)*\s*$/" tmp

    # remove whitespaces
    awk '{$1=$1};1' tmp > tmp3 && mv tmp3 tmp

    # remove empty lines
    sed -i -e '/^$/d' tmp

    paste -d '|' tmp2 tmp > "$1"

    rm tmp2 tmp
}

if [ -z "$1" ]; then
    echo "No corpus directory specified!"
    exit 1
fi

cd "$1"

find . -type f -name "transcript.txt" | while read file; do normalize "$file"; done