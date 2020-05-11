# textDMM 
experimenting with Deep Markov Models for character-level language modeling. 

You can find my full report ![here](/report/report.pdf).

The experiments were kind of meh, these DMMs are hard to stabilize during training so it's difficult to generate realistic text. It's decent, but doesn't quite compare to the results from [2], or even standard LSTMs, in terms of NLL loss.

## Requirements
```
python 3.6
pytorch 1.5.0
torchtext 0.6.0
pyro-ppl 1.3.1

```

## Basic Usage
To train an instance of a DMM with the default hyperparameters, run:
```
python3 main.py
```

## References
[1] “Deep markov model.” [Online]. Available: https://pyro.ai/examples/dmm.html

[2] R. G. Krishnan, U. Shalit, and D. Sontag, “Structured inference networks for nonlinear state space models,”
2016.

[3] Z. M. Ziegler and A. M. Rush, “Latent normalizing flows for discrete sequences,” 2019.

[4] I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. M. Botvinick, S. Mohamed, and A. Lerchner,
“beta-vae: Learning basic visual concepts with a constrained variational framework,” in ICLR, 2017.


