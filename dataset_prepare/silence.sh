#!/bin/sh

process_a_track()
{
    FILE=$1

    fn="${FILE##*/}"
    we="${fn%.*}"

    echo "Determining split points ... $FILE"

    truncate -s 0 "silence/$dn/$we.txt"
    ffmpeg -nostats -v repeat+info -y -i "$FILE" -af silencedetect=-23dB:d=0.1 -vn -sn -f s16le /dev/null 2>&1 |\
    grep '\[silencedetect.*silence_' | awk '{print $5}' | xargs -n2 | while read x y; do echo "$x $y" >> "silence/$dn/$we.txt"; done	
}

if [ -z "$1" ]; then
    echo "No corpus directory specified!"
    exit 1
fi

cd "$1"

mkdir -p silence

for D in sounds/*; do
    if [ -d "${D}" ]; then
	dn="${D##*/}"
	mkdir -p silence/"$dn"	
	N=8
        (
        for F in "${D}"/*; do
	    i=$((i+1)); i=$((i%N)); [ "$i" -eq 0 ] && wait
	    process_a_track "$F" &
	done	
        )
	
    fi 
done
