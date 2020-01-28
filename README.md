- [ ] Common Voice (de, ru, ~~es~~, fr, zh, nl)
  1. [x] remove invalid length/duration ratio
  2. [x] trim silence
  3. [x] apply noise filters
  4. [x] clear utterances (pinyin)
  5. [x] train deutsch multi-speaker (quality of CV speakers is really low, so it cannot be used to produce clearly intelligible speech, however the model converges and has the notion of different speakers :slightly_smiling_face:, a tiny speaker embedding like 16-32 with same params as mono-speaker is ok, 64 does not converge)
  6. [ ] train fr, ru, zh, nl multi-speaker models to verify the data
  
- [ ] Voice cloning
  1. [ ] use CV data to train bilingual model
  2. [ ] use CV data to train more languages
  3. [ ] implement the orthogonal adversarial loss? (probably not)
  
- [ ] Multi-lingual
  1. [x] replace BN with GN (does not converge even for LN or something up to 16 groups)
  2. [x] reducing encoder size (224 works, 208 converges slowly)
  3. [x] encoder bottleneck (reduction of the whole encoder works better)
  4. [x] encoder or context dropout (low rate diverges, higher rate does not converge)
  5. [x] multi-lingual convolutional (grouped) encoder (requires another sampler) implementation
  6. [ ] debug multi-lingual convolutional encoder and train fr-ge
  7. [x] encoder meta-generator (Conv1d and BN) implementation
  8. [ ] debug meta-generator and train fr-ge
  9. [ ] enable fine-tuning of pretrained decoder (to single language or any subset of languages/speakers)
  10. [ ] pretrain decoder with CV data 
  11. [ ] train for as many langs as possible (requires model parallel instead of data parallel):
     - [ ] simple (currently fr-ge)
     - [ ] shared (currently fr-ge)
     - [ ] separate (currently fr-ge)
     - [ ] generated 
  
- [ ] Fine-tuning
  1. [ ] create balanced and nested LJ speech datasets
  2. [ ] create some Czech data and process them as LJ speech
  3. [ ] try transfer-learning to EN and CZ of:
      - [ ] mono-lingual
      - [ ] simple
      - [ ] shared 
      - [ ] separate
      - [ ] generated 
  4. [ ] measure something based on training set size:
      - [ ] pronounciation error
      - [ ] word skipping
      - [ ] user preference
      - [ ] eval MCD
