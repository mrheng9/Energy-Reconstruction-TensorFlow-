# NOvA CAF HDF5 Implement Development  
This tutorial based on the server @muon-neutrino.ps.uci.edu  
Before this, read all papers professor send you to understand what you are doing.  

 ![image](https://user-images.githubusercontent.com/80438168/169384112-ba0c39ed-f50a-4a03-bf0f-301b3690cc56.png)
https://news.fnal.gov/2014/10/fermilabs-500-mile-neutrino-experiment-up-and-running/

## Git clone this repo(If you have your own way to check on the codes, please skip this step)  
1, Download git from here: https://git-scm.com/downloads  
2, Navigate to the directory you would like to keep the nova(e.g. $cd /you/projects/)  
3, Clone the repository into this directory on your local computer  
```$ git clone https://github.com/StellaMaUCI/nova.git```  
4, Check codes with your favorite editor or IDE(e.g. Vim, Emacs, VS Code, Pycharm etc.)  
If you use vi/vim for the first time, be sure to learn vimtutor for at least 15mins and remember quit without save command ```:q!```

## Connect to Server
1, To connect the server, you must be on the UCI network. You can access the network with a VPN if you aren't on campus, info available here: https://www.oit.uci.edu/services/security/vpn/  
2, Contact maintainer of the server to get an account on the server such as you@muon-neutrino.ps.uci.edu, and an initial password.  
3, Connect to the server with your initial password:  
```$ssh you@muon-neutrino.ps.uci.edu```  
4, Change your initial password and follow prompts:  
```$passwd```  
*Note: Be sure your password is safe and correct. Your account will be locked when you input wrong password twice.  
You are not entitled to run at root or sudo.

## Deploy the repo to the server under your directory
1, You need to be able to access github from the server. You can create an ssh key for them to identify you, by running:
```$ ssh-keygen -t ed25519 -C "your_email@example.com"```
with your own email. To get your created public key, now run
```$ cat ~/.ssh/id_ed25519.pub```
Now tell github your public key by going to github.com, account settings (in the top right), "SSH And GPG Keys", then "New SSH Key". Enter the public key you just got, and now github will know it's you.

2, Your directory path on the server is /home/you/, init your remote repo    
```$ git clone url_to_repo```
where url_to_repo is ```ssh://git@github.com/BenJarg/nova/```, or that replacing "BenJarg" with your own username if you forked your own version (don't worry about that point if you're just starting though). Now, this repo should appear as the directory "nova".

## Setup Environment  
### Add the following to your profile or `~/.bashrc`:
```
# NOvA repository
NOVA=/home/you/nova/
export PYTHONPATH=$PYTHONPATH:$NOVA
```

### In file `nova/config.py` change the line below to your directory as need
```
base = "/home/stella/nova"
```

### Artifacts
Stored files are written to `yourbase`:
```
-nova
    \
     -models
     -data
     -logs
     -tensorboard
```
where `models` contains Keras-model files, `data` contains raw and preprocessed data, and `logs` and `tensorboard` contain CSV log files and Tensorboard log-files, if recorded. Models are named with all passed arguments in the name and a date and time.

### Check if python3.6 and tensorflow work
```$python3.6
>>> import tensorflow as tf
>>>tf.__version__
'1.8.0'
```

## Training
### Run 
#### for NuE energy: (good start)
Our data directory on the server: /storage/data/train_data/nova_data/  
```
$mkdir -p yourbase/models/debug  
$cd <yourbase>  
$nohup python3.6 train.py --mode nue/electron --path <path to data folder> --name <pick a name> &

Tmux(Alternertive way of nohup): Connect to Server without piping off
For example:
$tmux new -s sname  # sname is session name. Windows from 0 in default
$tmux detach or ctrl+b  # detach this window and session
$tmux ls  # check all sessions
$tmux attach -t sname # attach my session
$tmux kill-session -t sname # kill my session
$tmux switch -t sname # switch my session
```  
For example(This process will take about 14-16hs):
```
$nohup python3.6 train.py --mode nue --path yourbase/data/FD-FluxSwap-FHC --name nue_train_output &
```  
Check running display information in nohup.out
Check training result in yourbase/data/FD-FluxSwap-FHC

#### for electron energy: 
Change --mode nue to --mode electron 

#### for weighted:
Add the `--weighted` flag to run weighted training. For example, to repeat the Prod5 training on the IGB machines use:
```
$nohup python3.6 train.py --mode nue --path yourbase/data/FD-FluxSwap-FHC --name fhc_train_electron_weighted --weighted
```

## Testing
To test the prod5 models from above:
```
$mkdir yourbase/predictions/
```  
In test.py,  Change save_path to yourbase/predictions/{}_{}.pkl
```$cd <yourbase>
$ nohup python3.6 test.py --modelpath <your training results> --mode <nue/electron> --path <training file folder> --name <pick_a_name> &
For example:
$ nohup python3.6 test.py --modelpath yourbase/models/debug/nue_train_output_num_train_samples_1434778__num_valid_samples_363799_ --mode nue --path /storage/data/train_data/nova_data/NOVA_MP5/FD-FluxSwap-FHC --name fhc_nue_test_unweighted &
```
Check running display information in nohup.out
Check training result in yourbase/predictions/{}_{}.pkl

## Plotting
### Run Jupyter notebook
$ssh -L 8080:localhost:8889 you@muon-neutrino.ps.uci.edu
$python3.6 -m jupyter notebook --no-browser --port=8889
In your browser, 
http://localhost:8080/

### Analysis test result in Jupyter
It’s necessary to understand how to use matplot library and scipy statistic tool  
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html?highlight=norm

#### Jupyter essential shortcuts:  
```Ctrl + Enter - executes the current cell  
Ctrl+S - saves the notebook and checkpoint  
Enter switch to edit mode  
h - it shows all keyboard shortcuts  
a -	above new cell  
b -	below new cell  
```

### The notebook `plots_nue_energy.ipynb` helps with making plots. 
Remotely, run `jupyter notebook --no-browser --port=8889`    
Locally, run `ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host`  
If you want to make a new plot, simply use `load(fname)` with the file-name outputted at the end of running `test.py` (above). Then copy the code from the existing plots to make a new plot. The loaded object has keys `y` for the true energies, `yhat` for predicted energies, and `resolution` for `yhat/y` 
