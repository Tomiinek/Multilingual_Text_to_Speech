
if [ -z "$1" ]; then
    echo "No corpus target directory specified!"
    exit 1
fi

mkdir -p "$1/transcripts" && cd "$1/transcripts"

filepairs=(
	'a_little_princess' 'http://www.gutenberg.org/cache/epub/146/pg146.txt'
	'a_room_with_a_view' 'http://www.gutenberg.org/files/2641/2641-0.txt'
	'black_beauty' 'http://www.gutenberg.org/files/271/271-0.txt'
	'carmilla' 'http://www.gutenberg.org/cache/epub/10007/pg10007.txt'
	'daisy_miller' 'http://www.gutenberg.org/files/208/208-0.txt'
	'emma' 'http://www.gutenberg.org/files/158/158-0.txt'
	'ethan_frome' 'http://www.gutenberg.org/files/4517/4517-0.txt'
	'far_from_the_madding_crowd' 'http://www.gutenberg.org/cache/epub/27/pg27.txt'
	'madame_de_treymes' 'http://www.gutenberg.org/cache/epub/4518/pg4518.txt'
	'mansfield_park' 'http://www.gutenberg.org/files/141/141-0.txt'
	'persuasion' 'http://www.gutenberg.org/cache/epub/105/pg105.txt'
	'pride_and_prejudice' 'http://www.gutenberg.org/files/1342/1342-0.txt'
	'sense_and_sensibility' 'http://www.gutenberg.org/cache/epub/161/pg161.txt'
	'silas_marner' 'http://www.gutenberg.org/cache/epub/550/pg550.txt'
	'summer' 'http://www.gutenberg.org/files/166/166-0.txt'
	'the_awakening' 'http://www.gutenberg.org/files/160/160-0.txt'
	'the_emerald_city_of_oz' 'http://www.gutenberg.org/cache/epub/517/pg517.txt'
	'the_gift_of_the_magi' 'http://www.gutenberg.org/cache/epub/7256/pg7256.txt'
	'the_jungle_books' 'http://www.gutenberg.org/cache/epub/35997/pg35997.txt'
	'the_patchwork_girl_of_oz' 'http://www.gutenberg.org/cache/epub/955/pg955.txt'
	'the_scarlet_letter' 'http://www.gutenberg.org/cache/epub/33/pg33.txt'
	'the_secret_garden' 'http://www.gutenberg.org/files/113/113-0.txt'
	'the_velveteen_rabbit' 'http://www.gutenberg.org/cache/epub/11757/pg11757.txt'
	'through_the_looking_glass' 'http://www.gutenberg.org/files/12/12-0.txt'
	'treasure_island' 'http://www.gutenberg.org/files/120/120-0.txt'
	'washington_square' 'http://www.gutenberg.org/files/2870/2870-0.txt'
)

for (( idx=0 ; idx<${#filepairs[@]} ; idx+=2 )) ; do
    
    LEFT=${filepairs[idx]}
    RIGHT=${filepairs[idx+1]}

    echo Downloading "$RIGHT" to "$LEFT"
    mkdir -p "$LEFT"
    cd "$LEFT"
    
    curl "$RIGHT" | tr -d '\r' > "00-full.txt"  

    echo Processing chapters of "$LEFT"
    awk -v RS="\n\n\n(\n)+" '{ print > ("chp." NR ".txt") }' "00-full.txt"

    for file in chp.*.txt; do
	sed -i -e 's/  \+/ /g' \
	    -e 's/Mrs\./misess/g' \
	    -e 's/Mr\./mister/g' \
	    -e 's/Dr\./doctor/g' \
	    -e 's/St\./saint/g' \
	    -e 's/Co\./company/g' \
	    -e 's/Jr\./junior/g' \
	    -e 's/Maj\./major/g' \
	    -e 's/Gen\./general/g' \
	    -e 's/Ft\./fort/g' \
	    -e 's/--/ -- /g' "$file"
	# etc.
	# ult.
	# By accident, I removed the lines which splitted sentences into words :(
	# this could be a good starting point:
	# | sed 's/\([.!?;]\) \([[:upper:]]\)/\1\n\2/g'
    done

    cd ../
     
done
