#!/bin/sh

process_a_track()
{
    FILE=$1

    fn="${FILE##*/}"
    we="${fn%.*}"
    ext="${F##*.}"

    align=silegnment/"$dn"/"$we".tsvm
    if [ ! -f "$align" ]; then
        continue
    fi
	
    nl=$(wc -l < "$align")
    echo "Splitting ${FILE} into ${nl} pieces"
    N=8
    (
    	while read -r start end uid; do	     
   	    i=$((i+1)); i=$((i%N)); [ "$i" -eq 0 ] && wait 
            ffmpeg -y -loglevel panic -i "$FILE" -ss "$start" -to "$end" "segments/$dn/$we-$uid.$ext" </dev/null &
    	done < "$align"	
    )
}


if [ -z "$1" ]; then
    echo "No corpus directory specified!"
    exit 1
fi

cd "$1"

mkdir -p segments

for D in sounds/*; do
    if [ -d "${D}" ]; then
	dn="${D##*/}"
	mkdir -p segments/"$dn"	

	for F in "${D}"/*; do
	    process_a_track "$F"
	done
   fi
done

