import os
import sys
import time
import shutil
import torch
import wandb
import subprocess
import json
import importlib
from algorithm import core
from env import ACEnv

####### Import config file for the algorithm #######
import config_SAC_lambda as c
####################################################
algorithm = importlib.import_module(f"algorithm.{c.algorithm}")


def main():
    ## Start with the files used in the pretraining (to use the same settings)
    if c.pretrain_kwargs["with_pretrain_files"]:
        subprocess.run(["python", c.pretrain_kwargs["pretrain_dir"] + "main.py", json.dumps(c.pretrain_kwargs), c.train_or_test])
        
    ## Start from the scratch
    else:
        if len(sys.argv) > 1 :
            with_pretrain_files = True
            pretrain_kwargs = json.loads(sys.argv[1])
            train_or_test = sys.argv[2]
        else:
            with_pretrain_files = c.pretrain_kwargs["with_pretrain_files"]
            pretrain_kwargs = c.pretrain_kwargs
            train_or_test = c.train_or_test

        ## Basic settings for the experiment (Change here if you want to change the name of the experiment)
        exp_name = "exp_" + str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))+"_"+c.exp_name+"_"+train_or_test
        wandb.init(project="your_project_name", name=exp_name)   # use wandb for logging
        
        torch.set_num_threads(torch.get_num_threads())
        config_dict = {name: getattr(c, name) for name in dir(c) if not name.startswith("__")}
        wandb.config.update(config_dict)
        file_save(c.log_dir, exp_name, c.file_dict, with_pretrain_files, pretrain_kwargs["pretrain_dir"])

        env_object = getattr(ACEnv, c.env_name)
        base_object = getattr(core, c.core)
        agent_object = getattr(algorithm, "Algo")
        agent = agent_object(env=env_object, base=base_object, **c.algo_kwargs, **c.env_kwargs, log_dir = c.log_dir + exp_name, pretrain_kwargs = pretrain_kwargs)

        if train_or_test == "train":
            agent.train()
        elif train_or_test == "test":
            agent.test()
            

def file_save(log_dir, exp_name, file_dict, with_pretrain_files, pretrain_dir = None):
    os.makedirs(log_dir + exp_name + "/ckpt", exist_ok=True)
    for key, value in file_dict.items():
        ## Copy the folder if the key of 'file_dict' in the config file contains 'folder'
        if "folder" in key:
            shutil.copytree(value, log_dir + exp_name + value[1:], dirs_exist_ok=True)
        
        ## Copy the file if the key of 'file_dict' in the config file contains 'file'
        elif "file" in key:
            os.makedirs(log_dir + exp_name + os.path.dirname(value[1:]), exist_ok=True)
            if with_pretrain_files:
                shutil.copy2(pretrain_dir + value[1:], log_dir + exp_name + value[1:])
            else:
                shutil.copy2(value, log_dir + exp_name + value[1:])
    
    
if __name__ == '__main__':
    main()