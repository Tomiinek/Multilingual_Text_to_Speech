- [ ] Common Voice (de, ru, es, fr, zh, nl)
  1. [ ] remove invalid length/duration ratio
  2. [x] trim silence
  3. [x] apply noise filters
  4. [ ] clear utterances
  5. [ ] train deutsch multi-speaker
  
- [ ] Voice cloning
  1. [ ] use CV data to train bilingual model
  2. [ ] use CV data to train more languages
  3. [ ] implement the orthogonal adversarial loss? (probably not)
  
- [ ] Multi-lingual
  1. [x] replace BN with GN (does not converge even for LN or something up to 16 groups)
  2. [x] reducing encoder size (224 works, 208 converges slowly)
  3. [x] encoder bottleneck (reducing of the whole encoder works better)
  4. [ ] encoder or context dropout
  5. [ ] implement multi-lingual convolutional (grouped) encoder (requires another sampler)
  6. [ ] implement encoder meta-generator (Conv1d and BN)
  7. [ ] enable fine-tuning of pretrained decoder (to single language or any subset of languages/speakers)
  8. [ ] pretrain decoder with CV data 
  9. [ ] train for as many langs as possible (requires model parallel instead of data parallel):
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
  4. measure something based on training set size:
      - [ ] pronounciation error
      - [ ] word skipping
      - [ ] user preference
      - [ ] eval MCD
