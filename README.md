# sad
Subtle artificial deception (SAD):  Masquerading as the Commander-in-Chief

## To Do
- let random-markov make random sentences.  L
- LSTM: Train on word to vec with embedding (preferred)  K
  - With twitter database embedding
- or: LSTM: Train on one-hot-encoding of words  K
- remove usernames and hyperlinks  R
- Create a random test-set (to not train on) ~300  R
- Set up online questionair  R, L
  - Including a prototype
- Get users (>20)  K, L, R


## Installation

To run markov:
Install spacy: `sudo pip install -U spacy`
Download English model: `python -m spacy download en`

## Word-level LSTM

### Parameters 
* Readout layer / predictions from closest word in embedding space
* Embeddings used / embedding depth
* Max sequence length 
* Learning rate
* Batch size
* Activation function
