# Czech Text to Speech

## Task list:

### Short-term

- [x] download available english datasets:
  - [x] LJ Speech - single speaker 24h, 1-10s, LibriVox, 25kHz
  - [x] Blizzard 2013 (aka Nancy Corpus) - single speaker 147h, 49 knih, emotive storytelling style, 44.1kHz, mp3
  - [x] VCTK - multispeaker (108), 44h, has leading and trailing silence, poor quality, 48kHz
  - [x] LibriTTS - 2456 speakers, 585h, ~50GB, 24kHz, maybe poor quality, see the papers folder for construction details
  - [ ] M-AI-Labs
  - [x] acquire Blizzard 2013
  - [x] alignment od Blizzard (aligner output combined with silence intervals)
  - [x] normalization of Blizzardu
  - [x] write loaders for the datasets
  - [x] make some stats like utterance lengths, phonemes, triples of phonemes ...
- [x] ~~Montreal Forced Aligner~~
  - [x] ~~install~~
  - [ ] ~~try alignment of Blizzard, does not work for me :sob: :sob: maybe can try on Windows OS~~
  - [ ] ~~try out Czech acustic model (we probably need a g2p dictionary :sob:)~~
- [x] Aeneas
  - [x] install
  - [x] try alignment of Blizzardu
  - [x] try on a short czech text (works out-of-box)
- [x] segmentation of audio according to aligner output and silence intervals
- [ ] read https://github.com/kastnerkyle/representation_mixing
- [ ] ~~Czech audiobooks~~
    - [ ] find them
    - [ ] alignment, normalization 
- [x] ~~crowdsourcing~~
- [x] make familiar with Griffin-Lim
- [x] Tacotron - explore some opensource implementations - Ito, r9y9 a NVIDIA
- [x] write own implementation of Tacotron (2)
    - [ ] Rectified Adam
    - [ ] Cyclic scheduler
    - [ ] Guided Attention Loss
    - [ ] bigger batch size during inference?
    - [ ] Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis
- [x] radio
- [ ] implement GMVAE-Tacotron2 paper
- [ ] english vanilla
- [ ] czech vanilla

### Long short-term

- [ ] is it real to train WeveNet / WaveRNN / WaveGlow?
- [ ] https://www.readbeyond.it/ebooks.html has some audiobooks

### Long-term

- [x] ~~Neural Speech Synthesis with Transformer Network! already implemented :sob:~~
- [ ] GST together with Toctron 2 with SOTA stuff, ... the available opensource implementation does not work
- [ ] GMVAE ... looks much more better than the GST papers 
- [ ] transfer learning from english Tacotron to other languages
- [ ] context, TTS of longer text and MOS of multiple sentences or paragraphs with interpunction ...
- [ ] It would be interesting to explore the trade-off between the number of mel frequency bins versus audio quality in future work.

## Notes:

### Datasets used in the GMVAE-Tacotron2 paper

For training data, we use 190 hours of American English speech, read by 22 different female speakers. Importantly, the 22 datasets include both expressive and non-expressive speech: to the expressive audiobook data from Section 4.1 (147 hours) we add 21 high-quality proprietary datasets, spoken with neutral prosody. These contain 8.7 hours of long-form news and web articles (20 speakers), and 34.2 hours of of assistant-style
speech (one speaker).

To evaluate the ability of GMVAE-Tacotron to model speaker variation and discover meaningful speaker clusters, we used a proprietary dataset of 385 hours of high-quality English speech from 84 professional voice talents with accents from the United States (US), Great Britain (GB), Australia (AU), and Singapore (SG).

A single speaker US English audiobook dataset of 147 hours, recorded by professional speaker, Catherine Byers, from the 2013 Blizzard Challenge (King & Karaiskos, 2013) is used for training. The data incorporated a wide range of prosody variation. We used an evaluation set of 150 audiobook sentences, including many long phrases.

To test the model’s ability to model a single speaker in a controlled environment, we utilize the Blizzard 2013 dataset [28], which consists of audiobook narration performed in a highly animated manner by a professional speaker. We use a 140 hour subset of this dataset for which we were able to find transcriptions, making the dataset also suitable for future text-to-speech experiments.

For single-speaker models, we use an expressive audiobook dataset consisting of 50,086 training utterances (36.5 hours) and 912 test utterances. Multi-speaker models are trained using data from 58 voice assistant-like speakers, consisting of 419,966 training utterances (327 hours). We evaluate on a 9-speaker subset of the multi-speaker test data, consisting of 1808 utterances


### Los problemos
- [ ] normalization of spectrograms, problems with L1 & L2
- [ ] in order to use GL, the preemphasis is really needed; WaveNet can make inversion of everything
- [ ] attention cumulative weights
- [ ] sharpening or windowing of the attention mechanism
- [ ] power or energy spectrograms?