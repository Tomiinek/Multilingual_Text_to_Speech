- [ ] Common Voice (de, ru, ~~es~~, fr, zh, nl)
  1. [x] remove invalid length/duration ratio
  2. [x] trim silence
  3. [x] apply noise filters
  4. [x] clear utterances (pinyin)
  5. [x] train deutsch multi-speaker (quality of CV speakers is really low, so it cannot be used to produce clearly intelligible speech, however the model converges and has the notion of different speakers :slightly_smiling_face:, a tiny speaker embedding like 16-32 with same params as mono-speaker is ok, 64 does not converge)
  6. [x] train ~~fr~~, ~~ge~~, ~~ru~~, ~~zh~~, ~~nl~~ multi-speaker models to verify the data (nl is very hard even with plain css10; russian has very poor generated spectrograms (even though original waveforms are ok) and the resulting sound is awful, however it can learn attention even for CV data; chinese converges slowly but works)
  
- [ ] Voice cloning
  1. [x] use CV data to train bilingual model (fr-ge works like a sharm without any special adversarial tasks etc.; cloning does not work at all without multi-speaker data)
  2. [x] make working Adversarial classifier (slightly improves voice cloning (speaker identity) while varying language; loss weight upper bound is 2.5 and lower which still makes a sense is about 0.25)
  2. [ ] use CV data to train more languages (fr-ge-zh)
     - [ ] simple (should fail)
     - [ ] separate without conv.
     - [ ] separate with conv.
     - [ ] generated
     - [ ] generated with adversarial speaker classifier
  5. [ ] implement the orthogonal adversarial loss? (probably not)
  
- [ ] Multi-lingual
  1. [x] replace BN with GN (does not converge even for LN or something up to 16 groups)
  2. [x] reducing encoder size (224 works, 208 converges slowly)
  3. [x] encoder bottleneck (reduction of the whole encoder works better)
  4. [x] encoder or context dropout (low rate diverges, higher rate does not converge)
  5. [x] multi-lingual convolutional (grouped) encoder (requires another sampler) implementation
  6. [x] debug multi-lingual convolutional encoder and train fr-ge
  7. [x] encoder meta-generator (Conv1d and BN) implementation
  8. [x] debug meta-generator and train fr-ge
  9. [ ] enable fine-tuning of pretrained decoder (to single language or any subset of languages/speakers)
  10. [x] pretrain decoder with CV data (but this will be probably useless) 
  11. [ ] train for as many langs as possible (requires model parallel instead of data parallel):
      - [ ] simple (should fail)
      - [ ] separate without conv.
      - [x] separate with conv. (fr-ge-zh, running fr-ge-zh-hu-es)
      - [ ] generated 
  
- [ ] ~~Fine-tuning~~
