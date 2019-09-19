#!/bin/sh

normalize()
{
  FILE=$1
  #remove line ids if any
  sed -i -e 's/.*|//' "$FILE"

  # trailing leadning whitespaces
  awk '{$1=$1};1' "$FILE" > tmp && mv tmp "$FILE"   
  # wrong wrap becuase of quotation marks (but short sentences can be on the same line: "Yes!" Said John.)
  # should be run iteratively for three times for example
  sed -i -e 's/\(.\{16,\}[.;?!]"\) \([A-Z].\{24,\}\)/\1\n\2/' "$FILE" 
  grep -Hn --color=always '.\{16,\}[.;?!]" [A-Z].\{24,\}' "$FILE" 
  # Yes; No; Oh!
  sed -i '/^\(Yes;|No;|Oh!\)$/N;s/\n/ /' "$FILE" 
  # common problems
  sed -i -e 's/Mrs\./misess/g' \
        -e 's/Mr\./mister/g' \
      	-e 's/Mrs\ /misess/g' \
      	-e 's/Mr\ /mister/g' \
      	-e 's/[“”]/"/g' \
      	-e 's/[’‘`]/\x27/g' \
     	-e 's/&c/et cetera/g' \
	-e 's/&/and/g' \
      	-e 's/—/ -- /g' \
      	-e 's/\]/)/g' \
	-e 's/\[/(/g' \
	-e 's/ô/o/g' \
	-e 's/æ/ae/g' \
	-e 's/þ/p/g' \
	-e 's/î/i/g' \
	-e 's/[éê]/e/g' \
	-e 's/[àâ]/a/g' "$FILE"
  # remove some common abbreviations (Mr., Mrs., ... ?)
  sed -i -e 's/etc\.$/et cetera./' "$FILE"
  sed -i -e 's/ult\.$/ultimate./' "$FILE"
  sed -i -e 's/etc\./et cetera/g' "$FILE"
  sed -i -e 's/ult\./ultimate/g' "$FILE"
  # remove double dash from the beginning and end
  sed -i -e 's/^-- //' \
         -e 's/ --$//' "$FILE"
  # add quotation missing marks (caused by splitting direct speech sentences)
  awk -v RS='"' 'NR%2 == 0 {gsub(/\n/,"\"\n\"")} {printf "%s%s", $0, RT}' "$FILE" > tmp && mv tmp "$FILE"
  # remove quotation marks around whole lines
  sed -i -e '/^["]\+[^"]*["]\+$/s/"//g' "$FILE" 
  sed -i -e '/^""[^"]\+$/s/"//g' "$FILE" 
  sed -i -e '/^"".\+$/s/""/"/g' "$FILE"
  sed -i -e '/^"$/d' "$FILE"
  # remove double dash from the beginning and end AGAIN!
  sed -i -e 's/^-- //' \
         -e 's/ --$//' "$FILE"
  # remove double quotation marks from the end
  sed -i -e '/^[^"]\+""$/s/"//g' "$FILE" 
  sed -i -e '/^.\+""$/s/""//g' "$FILE"
  # remove double dash with quotation marks at the beginning
  sed -i -e 's/^"-- "/"/g' "$FILE" 
  sed -i -e 's/^"-- /"/g' "$FILE"
  #remove empty lines
  sed -i -r '/^\s*$/d' "$FILE"
  # list remaining weird characters (should be only numbers, resolved manually)
  #grep -Hn --color=always "[^ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz \"(),.:;?\!'-]" "$FILE"

  # add line ids
  nl -nrz -w6 -s'|' "$FILE" > tmp && mv tmp "$FILE"
}

if [ -z "$1" ]; then
    echo "No corpus directory specified!"
    exit 1
fi

cd "$1/transcripts/"

find . -type f | while read file; do normalize "$file"; done


