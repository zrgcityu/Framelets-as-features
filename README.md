# Framelets_as_features
Python codes for generating 2-hop graph framelets on 3 datasets: amazon-ratings, minesweeper and tolokers, see [this paper](https://openreview.net/forum?id=tJbbQfw-5wv) "A critical look at the evaluation of GNNs under heterophily: Are we really making progress?" for dataset details. The packages required for running this repository are the same as those in [Graph-Involved-Frame](https://github.com/zrgcityu/Graph-Involved-Frame). Note that the version of scikit-network should be 0.28.3.  

The packages used in the conda environment for running [the implementation](https://github.com/yandex-research/heterophilous-graphs) are listed in the file "env.txt". Experiments were done with an RTX 3090 graphics card and 64 GB of RAM. Driver version: 470.239.06. CUDA version: 11.4. Ubuntu version: 20.04.5 LTS.

To generate framelets for a single dataset, run the python file "2_hop_framelets.py". See actual codes for details. The random seed we used were 42 (for Table 1) and 42, 43, 44, 45 (for Table 2).

The processed graphs are stored the folder "2_hop_adj" and framelets are stored in the folder "2_hop_frame". It will take some time when running for the first time, especially for the dense graph tolokers.

Once the framelets are generated, choose the numpy file that stores framelets with low or high variance and paste it at a proper place in [the implementation](https://github.com/yandex-research/heterophilous-graphs). See the file "example.txt" for an example on modifying the file "datasets.py" in the implementation. Note that for the dataset amazon-ratings, the first underscore "_" in the name of the numpy file should be replaced by a hyphen "-" before pasting.
