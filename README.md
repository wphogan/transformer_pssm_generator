# PSSM Generator
This repo contains code to train, test, and generate position-specific scoring matrices for protein sequences. It uses a transformer based on the transformer from [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Setup
1. Install packages with `pip install -r requirements.txt`
2. Download `cb513+profile_split1.npy.gz` (CB-513), `cullpdb+profile_5926.npy.gz` (TR-5534), and `cullpdb+profile_6133.npy.gz` (TR-6614) from [ICML2014](https://www.princeton.edu/~jzthree/datasets/ICML2014/). Place files in `data/`.

## Training
1. Run `preprocess/preprocess.py` to create JSON data files.
2. Run `main.py` to train the transformer and generate PSSMs. The uses TR-5534 for training and validation, and generates PSSMs for TR-6614.

## Generation
1. `generate.py` can be used to generate PSSMs using a pre-trained model. The current settings are to generate PSSMs for TR-6614
 
