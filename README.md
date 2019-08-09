# Czech Text to Speech

## Task list:

### Short-term

- [ ] stahnout vsechny anglicky datasety:
  - [x] LJ Speech        - single speaker 24h, 1-10s, LibriVox
  - [x] Blizzard 2013 (aka Nancy Corpus) - single speaker 147h, 49 knih, emotive storytelling style
  - [x] VCTK             - multispeaker (108), 44h, je treba osekat silence, vypada dobre
  - [x] LibriTTS      - 2456 speakers, 585h, moc velky, nestahnu
  - [ ] zjistit jak vypadaji, jak je nacitat, preprocessovat atd.
  - [ ] udelat si z nich statistiky delek promluv, fonemu, trojice fonemu atd.
- [ ] Montreal Forced Aligner --> alignment i grapheme to phoneme :)
  - [ ] nainstalovat a prozkoumat
  - [ ] vyzkouset na Blizzardu
  - [ ] vyzkouset predtrenovany cesky model na nejake knizce
- [ ] kouknout na https://github.com/kastnerkyle/representation_mixing
- [ ] ceske audioknihy
    - [ ] nejake najit
    - [ ] udelat alignment, normalizaci a prepsat je na fonemy (https://github.com/bootphon/phonemizer pokud by MFA neslo)
- [ ] napsat Mise a popr. udelat nejaky rozhrani jako ma Mozilla TTS 
- [ ] rozhlas

### Long short-term

- [ ] vyzkouset a zjistit jak funguje Griffin-Lim
- [ ] Tacostroj - kouknout na implementace - Ito, r9y9 a NVIDIA
- [ ] zjistit jestli je vubec realny natrenovat WeveNet / WaveRNN / WaveGlow
- [ ] vyzkouset vanilku na anglictine
- [ ] natrenovat vanilku s GL (na cestine)

### Long-term

- [ ] zajimavy by bylo prozkoumat transfer learning na jiny jazyk z anglickyho Tacostroje

## Notes:

### Jak maji vypadat datasety

Using Audio Books for Training a Text-to-Speech System

I am currently of the opinion that it isn't (solely) audio quality, rather some fundamental difference in the content / timing / delivery. Models seem extremely robust to recording noise (we used to add huge amounts of noise to the audio data itself in char2wav), but not to other simple issues. Even excessive leading / trailing silences can be an issue, or too short sentences. Talking to other people who do TTS they mention the same things, and it's really weird and mysterious to have a feel for what datasets will work, and what won't.

https://www.isca-speech.org/archive/interspeech_2010/i10_0158.html

https://www.researchgate.net/publication/4029874_Corpus_building_for_data-driven_TTS_systems


### Jake datasety

Blizzard 2013 dataset (Prahallad et al., 2013). For our experiments, we used a 20.5 hour subset of the dataset segmented into 9,741 utterances. We evaluated the model using the procedure described in Section 4.4, which encourages raters to compare synthesized audio directly with the ground truth. On the held out set, 16 kHz companded and expanded audio receives a MOS score of 4.65±0.13, while our synthesized audio received a MOS score of 2.67±0.37.

We train all the aforementioned models on the VCTK dataset with 44 hours of speech, which contains 108 speakers with approximately 400 utterances each. We also train all models on an internal dataset of audiobooks, which contains 477 speakers with 30 minutes of audio each (for a total of ∼238 hours).

Our second experiment uses real data. This dataset is made up of audio tracks mined from 439 official TED YouTube channel videos. The tracks contain significant acoustic variations, including channel variation (near- and far-field speech), noise (e.g. laughs), and reverberation. We use an endpointer to segment the audio tracks into short clips, followed by an ASR model to create <text, audio> training pairs. Despite the fact that the ASR model generates a significant number of transcription and misalignment errors, we perform no other preprocessing. The final training set is about 68 hours long and contains about 439 speakers.

Single-speaker dataset: A single speaker high-quality English dataset of audiobook recordings by Catherine Byers (the speaker from the 2013 Blizzard Challenge). This dataset consists of 147 hours of recordings of 49
books, read in an animated and emotive storytelling style.

Multi-speaker dataset: A proprietary high-quality English speech dataset consisting of 296 hours across 44 speakers (5 with Australian accents, 6 with British accents, 1 with an Indian accent, 2 with Singaporean accents, and 30 with United States accents).

We used two public datasets for training the speech synthesis and vocoder networks. VCTK [21] contains 44 hours of clean speech from 109 speakers, the majority of which have British accents. We downsampled the audio to 24 kHz, trimmed leading and trailing silence (reducing the median duration from 3.3 seconds to 1.8 seconds), and split into three subsets: train, validation (containing the same speakers as the train set) and test (containing 11 speakers held out from the train and validation sets). LibriSpeech [12] consists of the union of the two “clean” training sets, comprising 436 hours of speech from 1,172 speakers, sampled at 16 kHz. The majority of speech is US English, however since it is sourced from audio books, the tone and style of speech can differ significantly between utterances from the same speaker. We resegmented the data into shorter utterances by force aligning the audio to the transcript using an ASR model and breaking segments on silence, reducing the median duration from 14 to 5 seconds. As in the original dataset, there is no punctuation in transcripts. The speaker sets are completely disjoint among the train, validation, and test sets. Many recordings in the LibriSpeech clean corpus contain noticeable environmental and stationary background noise. We preprocessed the target spectrogram using a simple spectral subtraction denoising procedure, where the background noise spectrum of an utterance was estimated as the 10th percentile of the energy in each frequency band across the full signal. This process was only used on the synthesis target; the original noisy speech was passed to the speaker encoder.

The speaker encoder was trained on a proprietary voice search corpus containing 36M utterances with median duration of 3.9 seconds from 18K English speakers in the United States. This dataset is not transcribed, but contains anonymized speaker identities. It is never used to train synthesis networks.

For encoder conditioning, we used a neural network language model (NNLM) [14] trained on English Google News 200B corpus from TensorFlow Hub as the word embedding module. The module maps each word to a 128-dimensional vector. We also tried word2vec (W2V) [17] trained on the same corpus as the word embedding module. For decoder pre-training, we used VCTK [18], a publicly available corpus containing 44 hours of speech from 109 speakers, the majority of which have British accents. Note that there is an accent mismatch between the decoder pretraining (multiple speakers with British accents) and finetuning (single speaker with US accent) datasets. As mentioned above, we only use the speech signals in VCTK but not their transcripts.

We have shown that our framework makes end-to-end TTS feasible in small-data regime. Specifically, a semi-supervised trained Tacotron can produce intelligible speech using just 24 minutes of paired training data. This promising result also provides some guiding principles for future data collection efforts for both single and multi-speaker TTS. While we used Tacotron as the TTS model in this study, we believe the frame work is generally applicable to other end-to-end TTS models.

Lastly, since the main focus of this work is to make end-to-end TTS feasible in small-data regime instead of producing high-fidelity audio, we only used Griffin-Lim as the waveform synthesizer. To produce high-fidelity speech with very little paired data, we still need to address the problem of adapting neural vocoders in the semi-supervised setting.

For training data, we use 190 hours of American English speech, read by 22 different female speakers. Importantly, the 22 datasets include both expressive and non-expressive speech: to the expressive audiobook data from Section 4.1 (147 hours) we add 21 high-quality proprietary datasets, spoken with neutral prosody. These contain 8.7 hours of long-form news and web articles (20 speakers), and 34.2 hours of of assistant-style
speech (one speaker).

We train a WaveNet model for each of our three methods using the same dataset, which combines the high-quality LibriSpeech audiobook corpus (Panayotov et al., 2015) and a proprietary speech corpus. The LibriSpeech dataset consists of 2302 speakers from the train speaker subsets and approximately 500 hours of utterances, sampled at a frequency of 16 kHz. The proprietary speech corpus consists of 10 American English speakers and approximately 300 hours of utterances, and we down-sample the recording frequency to 16 kHz to match LibriSpeech.

To evaluate the ability of GMVAE-Tacotron to model speaker variation and discover meaningful speaker clusters, we used a proprietary dataset of 385 hours of high-quality English speech from 84 professional voice talents with accents from the United States (US), Great Britain (GB), Australia (AU), and Singapore (SG).

A single speaker US English audiobook dataset of 147 hours, recorded by professional speaker, Catherine Byers, from the 2013 Blizzard Challenge (King & Karaiskos, 2013) is used for training. The data incorporated a wide range of prosody variation. We used an evaluation set of 150 audiobook sentences, including many long phrases.

We used an audiobook dataset 2 derived from the same subset of LibriVox audiobooks used for the LibriSpeech corpus (Panayotov et al., 2015), but sampled at 24kHz and segmented differently, making it appropriate for TTS instead of speech recognition.

We utilize the VCTK speech synthesis dataset containing correlated speaker and background noise conditions for controlled experiments.

For all the experiments we trained on the LJ speech dataset [15]. This data set consists of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. The data consists of roughly 24 hours of speech data recorded on a MacBook Pro using its built-in microphone in a home environment. We use a sampling rate of 22,050kHz.

To test the model’s ability to model a single speaker in a controlled environment, we utilize the Blizzard 2013 dataset [28], which consists of audiobook narration performed in a highly animated manner by a professional speaker. We use a 140 hour subset of this dataset for which we were able to find transcriptions, making the dataset also suitable for future text-to-speech experiments.

Audiobook data is recorded in a highly controlled environment. To demonstrate MelNet’s capacity to model distributions with significantly more variation, we utilize the VoxCeleb2 dataset [8]. The VoxCeleb2 dataset consists of over 2,000 hours of speech data captured with real world noise including laughter, cross-talk, channel effects, music and other sounds. The dataset is also multilingual, with speech from speakers of 145 different nationalities, covering a wide range of accents, ages, ethnicities and languages.

We also train MelNet on a significantly more challenging multi-speaker dataset. The TED-LIUM 3 dataset [21] consists of 452 hours of recorded TED talks. The dataset has various characteristics that make it particularly challenging. Firstly, the transcriptions are unpunctuated, unnormalized, and contain errors.

For single-speaker models, we use an expressive audiobook dataset consisting of 50,086 training utterances (36.5 hours) and 912 test utterances. Multi-speaker models are trained using data from 58 voice assistant-like speakers, consisting of 419,966 training utterances (327 hours). We evaluate on a 9-speaker subset of the multi-speaker test data, consisting of 1808 utterances

We train models using a proprietary dataset composed of high quality speech in three languages: (1) 385 hours of English (EN) from 84 professional voice actors with accents from the United States, Great Britain, Australia, and Singapore; (2) 97 hours of Spanish (ES) from 3 female speakers include Castilian and US Spanish; (3) 68 hours of Mandarin (CN) from 5 speakers.