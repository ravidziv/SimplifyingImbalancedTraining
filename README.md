
# Simplifying Neural Network Training Under Class Imbalance

## Overview
This repository is an implementation of the methods and experiments presented in the NeurIPS 2023 paper "Simplifying Neural Network Training Under Class Imbalance". Our work focuses on addressing the challenges associated with training neural networks on datasets with class imbalances. This repository contains various modules and experiments that demonstrate our approach.

## Citation
If you find our work useful, please cite our paper:
```
@inproceedings{
shwartz-ziv2023simplifying,
title={Simplifying Neural Network Training Under Class Imbalance},
author={Ravid Shwartz-Ziv and Micah Goldblum and Yucen Lily Li and C. Bayan Bruss and Andrew Gordon Wilson},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=iGmDQn4CRj}
}
```

## Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/[YourGitHubUsername]/SimplifyingImbalancedTraining.git
cd SimplifyingImbalancedTraining
pip install -r requirements.txt
```

## Structure
The repository includes the following directories and files:
- `expriments/`: Contains scripts and configurations for experiments.
- `imbalanced/`: Modules related to handling class imbalance.
- `self_supervised/`: Self-supervised learning components.
- `requirements.txt`: Lists the dependencies for the project.
- `setup.py`: Setup script for the project.

## Usage
Navigate to the `expriments` directory to run specific experiments:
```bash
cd expriments
python [experiment_script.py] --options
```


##  Authors and Acknowledgments
This project is based on the work of Ravid Shwartz-Ziv, Micah Goldblum, Yucen Lily Li, C. Bayan Bruss, and Andrew Gordon Wilson. We extend our gratitude to New York University and Capital One for their support.

