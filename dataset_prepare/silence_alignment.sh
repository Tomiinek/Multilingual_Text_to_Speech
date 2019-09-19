#!/bin/sh

if [ -z "$1" ]; then
    echo "No corpus directory specified!"
    exit 1
fi

path=$(realpath silence_alignement.py)
cd "$1"

mkdir -p silegnment

for D in sounds/*; do
    if [ -d "${D}" ]; then
	dn="${D##*/}"
	mkdir -p silegnment/"$dn"	
	for F in "${D}"/*; do
	    fn="${F##*/}"
	    we="${fn%.*}"
	    echo "Refining alignment ... $F"

	    python "$path" -a "alignment/$dn/$we.tsvm" -s "silence/$dn/$we.txt" -o "silegnment/$dn/$we.tsvm"

	done
    fi
done
