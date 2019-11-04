#!/bin/sh

normalize()
{
  FILE=$1

  echo "Cleaning ... $1"

  # remove dots from the beginning
  sed -i -e 's/\([^|]*|[^|]*|\)\s*\.\+\s*/\1/' "$FILE"

  # remove double dash from the beginning and end
  sed -i -e 's/\([^|]*|[^|]*|\)\s*[–\-]\+\s*/\1/' \
         -e 's/\s*[–\-]\+\s*$//' "$FILE"

  # remove parenthessis from the beginning and end
  sed -i -e 's/\([^|]*|[^|]*|\)\s*[)(]\+\s*/\1/' \
	 -e 's/\s*[)(]\+\s*$//' "$FILE"

  # multiple dots at the end change to a single dot
  sed -i -e 's/\s*\.\+\s*$/\./' "$FILE"

  # multiple dots change to a single minus
  sed -i -e 's/\.\.\+/-/g' "$FILE"

  # remove redundant minus
  sed -i -e 's/\([?!\.:;]\s*\)[–\-]\+/\1/g' "$FILE"

  # trailing and leading whitespaces
  sed -i -e 's/\([^|]*|[^|]*|\)\s*/\1/' \
	 -e 's/\s*$//' "$FILE"

  # remove leading and trailing spaces
  awk '{$1=$1};1' "$FILE" > tmp && mv tmp "$FILE"

  # remove lines with chapter numbers like I., III. and so on
  sed -i -e '/^\([^|]*|[^|]*|\)\s*[IVXCDM]\+\.$/d' "$FILE"

  # remove utterances with numbers
  awk -F "|" '$3!~/[0-9]/' "$FILE" > tmp && mv tmp "$FILE"
}                                                 

if [ -z "$1" ]; then
    echo "No corpus directory specified!"
    exit 1
fi

cd "$1"

find . -type f -name "*.csv" | while read file; do normalize "$file"; done

mv de_DE/ de/
mv en_UK/ en-gb/
mv en_US/ en-us/
mkdir -p es/male/
mv es_ES/male/tux/ es/male/tux/
mv es_ES/ es-419/
mv fr_FR/ fr-fr/
mv it_IT/ it/
mv pl_PL/ pl/
mv ru_RU/ ru/
mv uk_UK/ ukr-Cyrl/
rm -rf ru/female/hajdurova/mnogo_shuma_iz_nichego