
if [ -z "$1" ]; then
    echo "No target directory specified!"
    exit 1
fi

mkdir -p "$1" && cd "$1"

languages=(
    'en'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/en.tar.gz'
    'de'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/de.tar.gz'
    'fr'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/fr.tar.gz'
	'rw'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/rw.tar.gz'
    'cy'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/cy.tar.gz'
    'br'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/br.tar.gz'
    'cv'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/cv.tar.gz'
    'tr'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/tr.tar.gz'
    'tt'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/tt.tar.gz'
    'ky'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/ky.tar.gz'
    'ga'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/ga-IE.tar.gz'
    'kab'   'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/kab.tar.gz'
    'ca'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/ca.tar.gz'
    'zh'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/zh-TW.tar.gz'
    'sl'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/sl.tar.gz'
    'it'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/it.tar.gz'
    'nl'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/nl.tar.gz'
    'cnh'   'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/cnh.tar.gz'
    'eo'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/eo.tar.gz'
    'et'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/et.tar.gz'
    'fa'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/fa.tar.gz'
    'eu'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/eu.tar.gz'
    'es'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/es.tar.gz'
    'zh'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/zh-CN.tar.gz'
    'mn'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/mn.tar.gz'
    'sah'   'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/sah.tar.gz'
    'dv'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/dv.tar.gz'
    'rw'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/rw.tar.gz'
    'sv'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/sv-SE.tar.gz'
    'ru'    'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/ru.tar.gz'
)

for (( idx=0 ; idx<${#languages[@]} ; idx+=2 )) ; do

    LANG=${languages[idx]}
    FILE=${languages[idx+1]}

    echo Downloading "$FILE" to "$LANG"
    mkdir -p "$LANG" && cd "$LANG"
    
    ZIPPED="$LANG.tar.gz"
    wget -q -O "$ZIPPED" "$FILE"  

    echo Extracting "$ZIPPED"

    tar xzf "$ZIPPED"
    ls *.tsv | grep -v ^validated.tsv | xargs rm
    awk -F"\t" '{if ($5 == 0) print $0}' validated.tsv > tmp && mv tmp validated.tsv
    
    awk -F"\t" '{print $2}' validated.tsv > valid_files.txt
    find clips -name "*.mp3" | grep -vFf valid_files.txt | xargs rm -rf
    
    rm valid_files.txt "$ZIPPED"

    #echo Converting "$LANG" to wavs
    sed -i -e 's/\.mp3/\.wav/g' valid_files.txt
    #cd clips
    #find . -type f -name "*.mp3" | while read i; do ffmpeg -loglevel panic -y -i $i $(basename $i).wav; done
    #cd ../

    cd ../
     
done
