# Environment Setup for WorldOnRails.jl

There are three steps in this environment setup guide:
* Setting up CARLA
* Setting up the dataset
* Setting up Julia environment

## CARLA
CARLA Leaderboard has a great tutorial on how to set up CARLA. You can find it [here](https://leaderboard.carla.org/get_started/). It is sufficient to go through the first part only. After that, you can run the script `~/carla/carla_evaluate.sh` to evaluate the baseline agent (a random agent for now). 

## Dataset
The dataset is shared by the authors and can be found in the original repository. I shared a tiny fraction of the original dataset along with its converted LMDB format [here](https://drive.google.com/file/d/1Vxi7aDYVqhjOK_nGGupzWn14DpQc_VBp/view?usp=sharing) (KU Members only for now). You can use the script `utils/DatasetToLMDB.py` to convert the rest of the dataset to LMDB format. 

## Setting up Julia
The repository has a `Project.toml` file which lists all the dependencies. To run PyJulia to evaluate in CARLA, you need to first follow the instructions in `Turn off compilation cache` in this [link](https://pyjulia.readthedocs.io/en/latest/troubleshooting.html). This is a temporary workaround and this part will be updated with instructions to create a custom build of Python. 
