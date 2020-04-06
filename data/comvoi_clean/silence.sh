#!/bin/sh

process_a_track()
{
    percent=true
    sound=0.15
    duration=0.1

    trim=false
    begin=0.0
    last=0.350

    silence=false
    offset=0.25
    duration=0.2
    decibel=23

    FILE=$1

    fn="${FILE##*/}"
    we="${fn%.*}"

    if [ "$percent" = true ]
    then
	sox "$FILE" c_$we.wav silence 1 $duration $sound% reverse silence 1 $duration $sound% reverse && mv c_$we.wav "$FILE"
    fi

    if [ "$trim" = true ]
    then
	sox "$FILE" b_$we.wav trim $begin -$last && mv b_$we.wav "$FILE"
    fi

    if [ "$silence" = true ]
    then

	    truncate -s 0 "silence/$we.txt"
	    ffmpeg -nostats -v repeat+info -y -i "$FILE" -af silencedetect=-${decibel}dB:d=$duration -vn -sn -f s16le /dev/null 2>&1 |\
	    grep '\[silencedetect.*silence_' | awk '{print $5}' | xargs -n2 | while read x y; do echo "$x $y" >> "silence/$we.txt"; done	

	    s=`cat "silence/$we.txt" | head -n1 | cut -d " " -f2`
	    e=`cat "silence/$we.txt" | tail -n1 | cut -d " " -f2`

	    if [ -z "$e" ]
	    then
		e=`cat "silence/$we.txt" | tail -n1 | cut -d " " -f1`
	    fi
	    
	    if [ -z "$s" ]
	    then
		start=0
	    else
		start=$(echo "$s-$offset" | bc | awk '{printf "%f", $0}')
		end=$(echo "$e+$offset" | bc | awk '{printf "%f", $0}')
		ffmpeg -y -loglevel panic -i "$FILE" -ss "$start" -to "$end" a_$we.wav  && mv a_$we.wav "$FILE" </dev/null &
	    fi
     fi
}

if [ -z "$1" ]; then
    echo "No corpus directory specified!"
    exit 1
fi

cd "$1"

#mkdir -p silence

N=8
(
    for F in *.wav; do
        i=$((i+1)); i=$((i%N)); [ "$i" -eq 0 ] && wait
        process_a_track "$F" &
    done
)
