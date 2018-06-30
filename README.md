# sad
Subtle artificial deception (SAD):  Masquerading as the Commander-in-Chief.

This repo contains code for generating fake tweets based on two different pre-processed datasets which are also included. It was used for a course project at Radboud University.

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
