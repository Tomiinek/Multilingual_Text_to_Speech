## Code Structure

```
Root
 ║
 ╠═ train.py                   <- main file used for training of models
 ╠═ synthesize.py              <- file for synthesis of spectrograms using checkpoints
 ╠═ gta.py                     <- script for generating ground-truth-aligned spectrograms
 ╠═ utils                
 ║   ├── __init__.py           <- various useful rutines: build model, move to GPU, mask by lengths        
 ║   ├── audio.py              <- functions for audio processing, e.g., loading, spectrograms, mfcc, ...  
 ║   ├── logging.py            <- Tensorboard logger, logs spectrograms, alignments, audios, texts, ...        
 ║   ├── samplers.py           <- batch samplers to produce balanced batches w.r.t languages, speakers                   
 ║   └── text.py               <- text rutines: conversion to IDs, punctuation stripping, phonemicization
 ╠═ params     
 ║   ├── params                <- definition of default hyperparameters and their description   
 ║   ├── singles               <- multiple files with parameter settings for training of monolingual models               
 ║   └── ...                   <- multiple files with parameter settings for training of multilingual models
 ╠═ notebooks                
 ║   ├── analyze.ipynb         <- basic dataset analysis for inspection of its properties and data distributions, with plots
 ║   ├── audio_test.ipynb      <- experiments with audio processing and synthesis
 ║   ├── encoder_analyze.ipynb   <- analysis of encoder outputs, speaker and language embeddings with plots
 ║   ├── code_switching_demo.ipynb   <- code-switching synthesis demo         
 ║   └── multi_training_demo.ipynb   <- multilingual training demo
 ╠═ modules                
 ║   ├── attention.py          <- attention modules: location-sensitive att., forward att., and base class  
 ║   ├── cbhg.py               <- CBGH module known from Tacotron 1 with simple highway layer (not convolutional)
 ║   ├── classifier.py         <- adversarial classifier with gradient reversal layer, cosine similarity classifier
 ║   ├── encoder.py            <- multiple encoder architectures: convolutional, recurrent, generated, separate, shared
 ║   ├── generated.py          <- meta-generated layers: 1d convolution, batch normalization
 ║   ├── layers.py             <- regularized LSTMs (dropout, zoneout), convolutional block and highway convolutional blocks
 ║   └── tacotron2.py          <- implementation of Tacotron 2 with all its modules and loss functions
 ╠═ evaluation                
 ║   ├── code-switched         <- code-switching evaluation sentences
 ║   ├── in-domain             <- in-domain (i.e., from CSS10) monolingual evaluation sentences
 ║   ├── out-domain            <- in-domain (i.e., from Wikipedia) monolingual evaluation sentences in ten languages
 ║   ├── asr_request.py        <- script for scraping transcription of given audios from Google Cloud ASR
 ║   ├── cer_computer.py       <- script for calculating character error rate between transcripts pairs
 ║   └── mcd_request.py        <- script for getting mel cepstral distortion between two spectrograms (includes DTW)
 ╠═ dataset_prepare                
 ║   ├── mecab_convertor.py    <- romanization of Japanese script
 ║   ├── pinyin_convertor.py   <- romanization of Chinese script
 ║   ├── normalize_comvoi.sh   <- basic shell script for downloading, extracting and cleaning od some Common Voice data
 ║   ├── normalize_css10.sh    <- set of regular expressions for cleaning CSS10 dataset transcripts  
 ║   └── normalize_mailabs.sh  <- probably not complete set of reg. exp.s for cleaning M-AILABS dataset transcripts
 ╠═ dataset
 ║   ├── dataset.py            <- TTS dataset, contains mel and linear spec., texts, phonemes, speaker and language IDs and a 
 ║   │                            function for generating proper meta-files and spectrograms for some datasets (see loaders.py) 
 ║   └── loaders.py            <- methods for loading popular TTS datasets into standardized python list (see dataset.py above)
 ╚═ data                
     ├── comvoi_clean 
     │    ├── all.txt          <- prepared meta-file for cleaned Common Voice dataset
     │    └── silence.sh       <- script for removal of leading or trailing "silence" of Common Voice audios
     ├── css10
     │    ├── train.txt        <- prepared meta-file for training set of cleaned CSS10 dataset
     │    └── val.txt          <- prepared meta-file for validation set of cleaned CSS10 dataset
     ├── css_comvoi 
     │    ├── train.txt        <- prepared meta-file for training set of dataset which is mixture of CSS10 and CV
     │    └── val.txt          <- prepared meta-file for validation set of dataset which is mixture of CSS10 and CV
     └── prepare_css_spectrograms.py    <- ad-hoc script for generating linear and mel spectrograms, see README.md 
    
```