# Learning Algebraic Multigrid using Graph Neural Networks
Code for reproducing the experimental results in our paper:
https://arxiv.org/abs/2003.05744

## Requirements
 * Python >= 3.6
 * Tensorflow >= 1.14
 * NumPy
 * PyAMG
 * Graph Nets: https://github.com/deepmind/graph_nets
 * MATLAB >= R2019a
 * MATLAB engine for Python: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
    * Requires modifying internals of Python library for efficient passing of NumPy arrays, as described here: https://stackoverflow.com/a/45290997
 * tqdm
 * Fire: https://github.com/google/python-fire
 * scikit-learn
 * Meshpy: https://documen.tician.de/meshpy/index.html
 

## Steps to run this code as is (on AWS)
- Create a CloudFormation stack with the [Matlab 2019b template](https://github.com/mathworks-ref-arch/matlab-on-aws/blob/master/releases/R2019b/README.md). Follow [this video tutorial](https://uk.mathworks.com/videos/how-to-run-matlab-in-the-cloud-with-amazon-web-services-1542634996553.html?requestedDomain=)
- Connect to the resulting EC2 instance with ssh with GUI formarding (`-X` or `-Y`)
- Activate Matlab by running its activation script (in `usr/local/matlab/bin/activate_matlab.sh`)
- Install [Matlab Engine for Python](https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) (upgrade `pip` if necessary). Fix the bug to pass Numpy arrays directly to Matlab from [here](https://stackoverflow.com/a/45290997) 
- Install all other dependencies (Pyamg, Fire, Scikit-Learn, Meshpy, etc.), always downgrade to the first version released in 2020 if errors occurre (`pip install xx==`)
- Install Nvidia drivers if not already present (check installation by running `nvidia-smi`). Refer to requirements files for version details 
- Install miniconda
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
- Create new conda environment. Use the `requirements.txt` file to replicate our environment
- Install [tensorflow 1.15](https://www.pugetsystems.com/labs/hpc/How-To-Install-TensorFlow-1-15-for-NVIDIA-RTX30-GPUs-without-docker-or-CUDA-install-2005/) from nvidia wheels with multi-gpu support 
- Downgrade tqdm to 4.40.2. In fact install all libs in `requirements_pip_in_conda.text` using pip


## Training
### Graph Laplacian
```
python train.py
```
Model checkpoint is saved at 'training_dir/*model_id*', where *model_id* is a randomly generated 5 digit string.

Tensorboard log files are outputted to 'tb_dir/*model_id*'.

A copy of the .py files and a JSON file that describes the configuration are saved to 'results/*model_id*'.

A random seed can be specified by setting a `-seed` argument.
### Spectral clustering
```
python train.py -config SPEC_CLUSTERING_TRAIN -eval-config SPEC_CLUSTERING_EVAL
```

### Ablation study
```
python train.py -config GRAPH_LAPLACIAN_ABLATION_MLP2
python train.py -config GRAPH_LAPLACIAN_ABLATION_MP2
python train.py -config GRAPH_LAPLACIAN_ABLATION_NO_CONCAT
python train.py -config GRAPH_LAPLACIAN_ABLATION_NO_INDICATORS
```
Other model configurations and hyper-parameters can be trained by creating `Config` objects in `configs.py`, and setting the appropriate `-config` argument.

## Evaluation
### Graph Laplacian lognormal distribution
```
python test_model.py -model-name 12345  
```
Replace `12345` by the *model_id* of a previously trained model.

Results are saved at 'results/*model_id*'.

### Graph Laplacian uniform distribution
```
python test_model.py -model-name 12345 -config GRAPH_LAPLACIAN_UNIFORM_TEST
```

### Finite element
```
python test_model.py -model-name 12345 -config FINITE_ELEMENT_TEST
```

### Spectral clustering
```
python spec_cluster.py -model-name 12345
```