#!/bin/sh

if [ -z "$1" ]; then
    echo "No corpus directory specified!"
    exit 1
fi

cd "$1"

# https://www.readbeyond.it/aeneas/docs/syncmap.html

mkdir -p alignment

for D in transcripts/*; do
    if [ -d "${D}" ]; then

	dn="${D##*/}"
	mkdir -p alignment/"$dn"
	for F in "${D}"/*; do
	  fn="${F##*/}"
	  we="${fn%.*}"
	  python -m aeneas.tools.execute_task \
		sounds/"$dn"/"$we".wav \
		transcripts/"$dn"/"$fn" \
		"task_language=eng|os_task_file_format=tsvm|is_text_type=parsed|is_audio_file_detect_head_max=120|is_audio_file_detect_head_min=0|is_audio_file_detect_tail_max=45|is_audio_file_detect_tail_min=0|os_task_file_head_tail_format=hidden" \
		alignment/"$dn"/"$we".tsvm
	done
    fi
done




