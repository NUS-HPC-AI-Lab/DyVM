# Ablation Study

We provide the code of each ablation study in the `Albatation` directory. You can follow the scripts in their `vim/scripts/` directory to reproduce the results.

## Token Only

Code that applies token pruning only.

## Block Only

Code that applies block pruning only.

## Pruning_Strategy

We test different pruning strategies, including random pruning and pruning with even distribution.

## Predictor_Input

We use the output of mamba block as the input of the predictor, including $B$ and $\Delta$.

