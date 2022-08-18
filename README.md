# L2G2L: a Scalable Local-to-Global Network Embedding with Graph Autoencoders

INTRO:
This project would provide codes for Local2GAE2Global (L2G2G) model, which is a scalable GAE approach combing the idea of local2global.

For comparison, I've also provied the codes for GAE and GAE+L2G, which is based on or refers to https://github.com/LJeub/Local2Global.

The repo is structed in the following way:
  L2G2L - folder for all the codes(following ones are the main files in this folder):
    1. run_LargeScale.py - running for L2G2G and GAE+L2G
    2. main.py - running for fastgae and gae for comparison
    3. model.py - includes the model classes we defined: L2G2G and GAE+Loc2Glob
    4. utils.py - includes the tools we need in processing


REQUIREMENTS:
  1. pytorch 1.11.0+cuda
  2. pytorch_geometric
  3. local2global library: https://github.com/LJeub/Local2Global


EXAMPLES:
  1. run test for L2G2G:
         python run_LargeScale.py --dataset='cora' --device=cuda --model=fgae --epoch=200 --lr=0.001 --num_patches=10
  2. run test for FastGAE:
         python main.py --dataset='cora' --device=cuda --epoch=200 --lr=0.001


REFERENCES:

[1] L. G. S. Jeub, G. Colavizza, X. Dong, M. Bazzi, M. Cucuringu (2021). Local2Global: Scaling global representation learning on graphs via local training. DLG-KDD'21.   arXiv:2107.12224 [cs.LG]

[2] M. Cucuringu, Y. Lipman, A. Singer (2012). Sensor network localization by eigenvector synchronization over the euclidean group. ACM Transactions on Sensor Networks 8.3. DOI: 10.1145/2240092.2240093

[3] Salha, Guillaume, et al. "FastGAE: Scalable graph autoencoders with stochastic subgraph decoding." Neural Networks 142 (2021): 1-19.

[4] Kipf, Thomas N., and Max Welling. "Variational graph auto-encoders." arXiv preprint arXiv:1611.07308 (2016).
