# Hyperparameter Tuning Using Parameter-Sherpa
These are basic instructions and notes to aid in using the sherpa library to tune models in the server @muon-neutrino.ps.uci.edu

## Python Virtual Environment
Create a Python virtual environment and pip install the necessary libraries inside your venv. 

### Creating a Virtual Environment
These are instructions to create the virtual environment
- Inside the directory you want the venv to be, use `python3.6 -m venv [virtualenvname] --system-site-packages`
- '-m' tells Python to use the 'venv' package
- '--system-site-packages' is an optional command that tells 'venv' to give the new virtual environment read and use access (but no edit) to the system's python packages
- If you want to know more about this command see https://stackoverflow.com/questions/43069780/how-to-create-virtual-env-with-python3 and https://docs.python.org/3/library/venv.html.

### Using the Virtual Environment
The virtual environment must be activated for you to use it and install new packages. 
- To activate and use the virtual environment, use `cd baseDirectory/bin/` and `source activate`
- You can then run commands and train models using the virtual env
- To deactivate from anywhere, use `deactivate`

### Installing Packages 
- A basic template for pip installation: `$ python3.6 -m pip install <library>`
Installing Sherpa:
- Make sure you install the correct Sherpa library as there are two called Sherpa: `parameter-sherpa`. Use command `$ python3.6 -m pip install parameter-sherpa`
Fixing other packages:
- The system's version of SciPy is incompatible with parameter-sherpa so we need to downgrade SciPy to version 1.4.1. While the virtual environment is activated, use command `python3.6 -m pip install scipy==1.4.1`

## Running Sherpa: 
There are multiple methods to actually run Sherpa. Please use the Sherpa Documentation and examples below to guide your code. You can either use `sherpa.Study()` or `sherpa.Client()` to automatically test sets of hyperparameters. 
- Method 1: Use `train_sherpa_tutorial.py`, `train_sherpa_gridsearch.py`, or make a copy of `train.py` and retrofit it to use `sherpa.Study()`
- Method 2: Use the files in `nova/sherpa` such as `runner_architecture.py`. These files use `sherpa.Client()`
- Other Methods: You can manually tuning hyperparameters or use other hyperparameter tuning libraries.

Useful Sherpa documentation: https://parameter-sherpa.readthedocs.io/en/latest/index.html <br />
Sherpa Examples: https://github.com/sherpa-ai/sherpa/tree/master/examples

### Method 1:
The two files `train_sherpa_tutorial.py` and `train_sherpa_gridsearch.py` are modified versions of `train.py` that run `sherpa.Study()`. To modify the file yourself, follow the sherpa documentation "A Guide to Sherpa" which explains the different objects like parameters, algorithms, and study. It can be run using a similar command as used for `train.py`: 
```
$ nohup python3.6 train_sherpa_tutorial.py --mode nue --path yourbase/data/FD-FluxSwap-FHC --name nue_train_output &
```
The two files are different in which algorithms they implement - bayesian optimization and grid search respectively. 

### Algorithms:
Bayesian Optimization:
- Description: https://parameter-sherpa.readthedocs.io/en/latest/algorithms/gpyopt.html
- Best for testing continuous parameters like momentum, learning rate, and regularizers.
- Time: 30 Trials run on 10 data files takes about 2 hours (with early stopping from regression.py active)

Grid Search:
- Description: "Explores a grid across the hyperparameter space such that every pairing is evaluated." It can use a list of `parameter` objects you created yourself as seen in `train_sherpa_gridsearch.py`, or you can use the method described in the documentation.
- Best for testing discrete parameters like number of layers and number of nodes per layer.
- Tip: it will create points to be tested for all values of an ordinal type parameter. Also, it is best to do this multiple times to see if results are consistent. 
- Time: depends on number of points.
  
There are other optimization algorithms besides bayesian_optimization and grid_search. To implement, refer to the documentation. <br />
	
## Trouble Shooting:
### Common Errors:
- NameError means you did not import the library in the file
- AttributeError likely means you installed the wrong library or there is another module or library with the same name
### Tips:
- To make training run faster, reduce the amount of data the file trains on. 
- Make sure you are in the virtual environment and the libraries are installed
- Try importing the library yourself inside the Python console to validate it works.
- The global tensorflow works much faster than installing it yourself in the venv. Tensorflow installed in the venv for some reason doesn't find the GPUs. 
- If you run into a new error and fix, please add the information here. 

