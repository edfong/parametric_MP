# Parametric Martingale Posteriors
This repository contains the code for the illustrations in the preprint "Asymptotics for parametric martingale posteriors" by Edwin Fong & Andrew Yiu which can be found [here](https://arxiv.org/abs/???). 

## Running Experiments
All experiment can be found in ``./run_scripts``, which utilize the functions implemented in ``./src``.

The Julia scripts can be run in terminal, e.g. with:
```
julia run_expon.jl
```

Outputs from the experiments are stored in ``./results``, and all plots can be produced by the Jupyter notebook ``./plots/plots.ipynb``. 

## Real Data Example

The notebook ``./data/load_data.ipynb`` downloads, preprocesses and saves the ACTG 175 dataset [Hammer
et al. (1996)] into ``./data``. 

## References
Hammer, S. M., Katzenstein, D. A., Hughes, M. D., Gundacker, H., Schooley, R. T., Haubrich, R. H.,Henry, W. K., Lederman, M. M., Phair, J. P., Niu, M., et al. (1996). A trial comparing nucleoside monotherapy with combination therapy in HIV-infected adults with CD4 cell counts from 200 to 500 per cubic millimeter. New England Journal of Medicine, 335(15):1081–1090.
