import os
from glob import glob
import sys


def get_loader_by_name(name):
    """Return the respective loading function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


def vctk(root_path, meta_files=None):
    """Load VCTK sound and meta files."""
    if meta_files is None: meta_files = glob(f"{root_path}/txt/**/*.txt", recursive=True)
    meta_files.sort()
    items = []
    language = ""
    for meta_file in meta_files:
        file_name = os.path.basename(meta_file).split('.')[0]
        speaker_name = file_name.split('_')[0]
        utterance_id = file_name.split('_')[1]
        with open(meta_file, 'r', encoding='utf-8') as ttf:
            text = ttf.read()        
            audio = os.path.join("wav48", speaker_name, file_name + ".wav")
            full_audio = os.path.join(root_path, audio)
            assert os.path.isfile(full_audio), (
                f'Referenced audio file {full_audio} does not exist!')  
            items.append([text[:-1], audio, speaker_name, language])
    return items


def mailabs(root_path, meta_files=None):
    """Load M-AILABS sound and meta files."""
    if meta_files is None: meta_files = glob(f"{root_path}/*/*/*/*/metadata.csv", recursive=True)
    meta_files.sort()
    items = []
    for meta_file in meta_files:
        book_dir = os.path.dirname(meta_file)
        speaker_dir = os.path.dirname(book_dir)
        language_dir = os.path.dirname(os.path.dirname(speaker_dir))
        with open(meta_file, 'r', encoding='utf-8') as ttf:
            speaker_name = os.path.basename(speaker_dir)
            language = os.path.basename(language_dir)  
            for line in ttf:         
                cols = line[:-1].split('|')    
                audio = os.path.join(book_dir[len(root_path)+1:], "wavs", cols[0] + ".wav")
                full_audio = os.path.join(root_path, audio)
                assert os.path.isfile(full_audio), (
                    f'Referenced audio file {full_audio} does not exist!')  
                items.append([cols[2], audio, speaker_name, language])
    return items


def css10(root_path, meta_files=None):
    """Load CSS10 sound and meta files."""
    if meta_files is None: meta_files = glob(f"{root_path}/*/transcript.txt", recursive=True)
    meta_files.sort()
    items = []
    for meta_file in meta_files:
        language_dir = os.path.dirname(meta_file)
        with open(meta_file, 'r', encoding='utf-8') as ttf:
            language = os.path.basename(language_dir) 
            speaker_name = language 
            for line in ttf:         
                cols = line.rstrip().split('|')    
                audio = os.path.join(language, cols[0])
                full_audio = os.path.join(root_path, audio)
                assert os.path.isfile(full_audio), (
                    f'Referenced audio file {full_audio} does not exist!')  
                items.append([cols[2], audio, speaker_name, language])
    return items


def my_blizzard(root_path, meta_files=None):
    """Load My Blizzard 2013 audio and meta files."""
    if meta_files is None: transcript_files = glob(f"{root_path}/transcripts/**/*.txt", recursive=False)     
    else: transcript_files = meta_files
    transcript_files.sort()
    folders = [os.path.dirname(f) for f in transcript_files]
    items = []
    speaker_name = ""
    language = ""
    for idx, transcript in enumerate(transcript_files):
        folder = folders[idx]        
        filename = os.path.splitext(os.path.basename(transcript))[0]
        with open(transcript, 'r', encoding='utf-8') as ttf:
            for line in ttf:
                cols = line[:-1].split('|')
                segments_folder = folder.replace(f"{root_path}/transcripts", "segments")
                audio = os.path.join(segments_folder, filename + '-' + cols[0] + '.wav')
                full_audio = os.path.join(root_path, audio)
                text = cols[1]
                assert os.path.isfile(full_audio), (
                    f'Referenced audio file {full_audio} does not exist!')  
                items.append([text, audio, speaker_name, language])
    return items


def ljspeech(root_path, meta_file=None):
    """Load the LJ Speech audios and meta files"""
    if meta_file is None: txt_file = os.path.join(root_path, "metadata.csv")
    assert os.path.isfile(txt_file), (f'Dataset meta-file not found: given given {txt_file}')  
    items = []
    speaker_name = ""
    language = ""
    with open(txt_file, 'r', encoding='utf-8') as ttf:
        for line in ttf:
            cols = line[:-1].split('|')
            audio = os.path.join('wavs', cols[0] + '.wav')
            full_audio = os.path.join(root_path, audio)
            text = cols[2]
            assert os.path.isfile(full_audio), (
                    f'Referenced audio file {full_audio} does not exist!')  
            items.append([text, audio, speaker_name, language])
    return items


def my_common_voice(root_path, meta_files=None):
    """Load My Common Voice sound and meta files."""
    if meta_files is None: meta_files = glob(f"{root_path}/*/meta.csv", recursive=True)
    meta_files.sort()
    items = []
    for meta_file in meta_files:
        language_dir = os.path.dirname(meta_file)
        with open(meta_file, 'r', encoding='utf-8') as ttf:
            language = os.path.basename(language_dir)  
            for line in ttf:       
                cols = line.rstrip().split('|')
                speaker_name = cols[0]  
                audio = os.path.join(language, "wavs", cols[0], cols[1])
                full_audio = os.path.join(root_path, audio)
                assert os.path.isfile(full_audio), (
                    f'Referenced audio file {full_audio} does not exist!')
                items.append([cols[2], audio, speaker_name, language])
    return items