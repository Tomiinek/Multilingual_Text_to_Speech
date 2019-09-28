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
    for meta_file in meta_files:
        file_name = os.path.basename(meta_file).split('.')[0]
        speaker_name = file_name.split('_')[0]
        utterance_id = file_name.split('_')[1]
        with open(meta_file, 'r') as ttf:
            text = ttf.read()        
            audio = os.path.join(root_path, "wav48", speaker_name, file_name + ".wav")
            if os.path.isfile(audio): items.append([text, audio, speaker_name])
            else: raise RuntimeError("> File %s does not exist!"%(audio))
    return items


def my_blizzard(root_path, meta_files=None):
    """Load My Blizzard 20013 audio and meta files."""
    if meta_files is None: transcript_files = glob(f"{root_path}/transcripts/**/*.txt", recursive=False)     
    else: transcript_files = meta_files
    transcript_files.sort()
    folders = [os.path.dirname(f) for f in transcript_files]
    items = []
    speaker_name = ""
    for idx, transcript in enumerate(transcript_files):
        folder = folders[idx]        
        filename = os.path.splitext(os.path.basename(transcript))[0]
        with open(transcript, 'r') as ttf:
            for line in ttf:
                cols = line.split('|')
                segments_folder = folder.replace("transcripts", "segments")
                audio = os.path.join(segments_folder, filename + '-' + cols[0] + '.wav')
                text = cols[1]
                if os.path.isfile(audio): items.append([text, audio, speaker_name])
                else: raise RuntimeError("> File %s does not exist!"%(audio))
    return items


def ljspeech(root_path, meta_file=None):
    """Load the LJ Speech audios and meta files"""
    if meta_file is None: txt_file = os.path.join(root_path, "metadata.csv")
    assert os.path.isfile(txt_file), (f'Dataset meta-file not found: given given {txt_file}')  
    items = []
    speaker_name = ""
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line[:-1].split('|')
            audio = os.path.join('wavs', cols[0] + '.wav')
            full_audio = os.path.join(root_path, audio)
            text = cols[2]
            if os.path.isfile(full_audio): items.append([text, audio, speaker_name])
            else: raise RuntimeError("> File %s does not exist!"%(full_audio))  
    return items
