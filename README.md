# Deep-Unfolded-NMF-for-Speech-Source-Separation

This project is an implementation of the paper "Deep Recurrent NMF for Speech Separation by Unfolding Iterative Thresholding" of the authors : Scott Wisdom, Thomas Powers, James Pitton and Les Atlas. The original paper code has two major constraints :

* It was developped using both Python and uses API calls to Matlab.
* The code is highly dependent on the private dataset CHiME2.

In this project we propose a full implementation in Python 3 (No Matlab) that can be used with wav files (more general).


We have updated the original algorithm in different ways to increase the temporal consistency of the audio data, the description in details is given in the final report. This approach allows a better generelization and interpretation of what the model is performing compared to SOTA deep learning.

![](/Images/method.jpg)

Here we can observe the loss evolution and the increases of SDR in both speech and music.

<img width="1099" alt="Capture d’écran 2022-07-31 à 5 32 59 PM" src="https://user-images.githubusercontent.com/55285736/182033829-fafa5739-4828-41ab-9093-98daf5b0d13e.png">

@article{DBLP:journals/corr/abs-1709-07124,
  author    = {Scott Wisdom and
               Thomas Powers and
               James W. Pitton and
               Les Atlas},
  title     = {Deep Recurrent {NMF} for Speech Separation by Unfolding Iterative
               Thresholding},
  journal   = {CoRR},
  volume    = {abs/1709.07124},
  year      = {2017},
  url       = {http://arxiv.org/abs/1709.07124},
  eprinttype = {arXiv},
  eprint    = {1709.07124},
  timestamp = {Mon, 13 Aug 2018 16:48:55 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1709-07124.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
