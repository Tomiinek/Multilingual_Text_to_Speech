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
    - [x] guided attention -- Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
    - [x] Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis
- [x] radio
- [ ] ~~implement GMVAE-Tacotron2 paper~~
- [x] english vanilla
- [ ] ~~czech vanilla~~
- [ ] optimize Tacotron parameters
- [ ] learn tacotron for each M-AILABs language

### Long short-term

- [ ] create multi-lingual baseline
- [ ] define and automatize experiments
- [ ] optimize experiments on baseline

### Long-term

- [ ] create more advanced multi-lingual models

### Los problemos
- [ ] normalization of spectrograms, problems with L1 & L2
- [ ] in order to use GL, the preemphasis is really needed; WaveNet can make inversion of everything
- [ ] attention cumulative weights
- [ ] sharpening or windowing of the attention mechanism
- [ ] power or energy spectrograms?