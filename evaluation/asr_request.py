from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from google.protobuf.json_format import MessageToJson, MessageToDict
import io
import os
import json

"""

**************************************** INSTRUCTIONS ***************************************
*                                                                                           *
*   Usage: python asr_request.py --language german --model ground-truth                     *
*                                                                                           *
*   For each audio file in model's folder request Google Cloud ASR for transcription and    *
*   save it into a file in the model/asr directory.                                         *
*                                                                                           *
*********************************************************************************************

"""

def sample_recognize(path, language_code, sample_rate_hertz):
    
    client = speech_v1.SpeechClient()

    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
    }

    with io.open(path, "rb") as f:
        content = f.read()
    audio = {"content": content}

    response = client.recognize(config, audio)
    response = MessageToDict(response, preserving_proto_field_name=True)
    
    if "results" not in response:
        return None

    return response["results"]


if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="Language to be synthesized.")
    parser.add_argument("--model", type=str, required=True, help="Model specific folder.")
    parser.add_argument("--where", type=str, required=True, help="Data specific folder.")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate.")
    args = parser.parse_args()

    language_mapping = {
        'dutch'		:	'nl-NL',	
        'finnish' 	:	'fi-FI',
        'french'	:	'fr-FR',
        'german'	:	'de-DE',
        'greek'		:	'el-GR',
        'hungarian'	:	'hu-HU',
        'chinese'	:	'zh',
        'japanese'	:	'ja-JP',
        'russian'	:	'ru-RU',
        'spanish'	:	'es-ES'
    }

    meta_file = os.path.join(args.where, f'{args.language}.txt')
    with open(meta_file, 'r', encoding='utf-8') as f:
        for l in f:
			
            tokens = l.rstrip().split('|')
            idx = tokens[0]

            sound_path = os.path.join(args.where, args.model, 'audios', args.language, f'{idx}.wav')		
            if not os.path.exists(sound_path): 
                continue
			
            asr_result = sample_recognize(sound_path, language_mapping[args.language], args.sample_rate)
            
            if asr_result is None:
                continue
			
            output_path = os.path.join(args.where, args.model, 'asr', args.language)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_file = os.path.join(output_path, f'{idx}.json')
            with open(output_file, 'w+', encoding='utf-8') as of:
                print(asr_result, file=of)