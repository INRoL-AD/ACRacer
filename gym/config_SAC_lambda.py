note = "S67-SAC_lambda-Run1-mainPC"
exp_name = "SAC_lambda"

env_name = "ACEnvSingle"
core = "MLPActorCritic"
algorithm = "SAC_lambda"
train_or_test = "train"

pretrain_kwargs = {
    "pretrain_dir": None,                       # directory of the pretrained files. if None, start from the scratch.
    "pretrain_ckpt_name": "your_ckpt_name",     # name of the pretrained model.
    "with_pretrain_files": False,               # if True, use the files used for the pretraining stage.
    "with_pretrain_buffers": False,             # if True, use the buffers used for the pretraining stage.
    "init_buffer_with_demo": True,              # if True, initialize the buffer with the demonstrations.
    "demo_ckpt_name": "your_demo_name"          # directory of the demonstration file.
}

env_kwargs = {
                "is_random_spawn": False,       # if True, the car will be spawned at random position.
                "random_spawn_portion": 0.8,    # if 1.0, fully random and if 0.0, fully deterministic.

                "frame_exist": False,           # if True, the frame information will be used.
                "track_exist": True,            # if True, the track information will be used.
                "preview_kwargs": {             # set parameters for the preview information (forward).
                    "mode":'static',            # 'static' or 'dynamic'. 
                    "num_sample": 30,           # number of samples in the preview information.
                    "sample_interval": 2,       # interval between samples [m].
                    "factor": 1.0,              # the factor for proportion btw velocity and preview distance (only for dynamic mode).
                },
                "distance_kwargs": {            # set parameters for the distance information (forward+backward).
                    "mode":'static',            # 'static' or 'dynamic'.
                    "num_sample": 250,          # number of samples in the distance information.
                    "num_sample_prev": 50,      # number of samples in the forward information. (remaining samples are assigned to backward)
                    "sample_interval": 1,       # interval between samples [m].
                    "factor": None,             # the factor for proportion btw velocity and preview distance (only for dynamic mode).
                },
                "lidar_2d_kwargs":{             # set parameters for the 2D rangefinder information.
                    "num_rays": 21,             # number of rays.
                    "distance_max": 200,        # maximum distance [m].
                    "roi_deg_min": -100,        # minimum angle of the field of view [deg].
                    "roi_deg_max": 100,         # maximum angle of the field of view [deg].
                }
}

algo_kwargs = { 
                "seed": 0,                          # random seed.
                "steps_per_epoch": 20000,           # number of steps per epoch.
                "epochs": 100,                      # number of epochs.

                "replay_size": int(1e6),            # size of the replay buffer.
                "batch_size": 1024,                 # batch size for the replay buffer.
                "batch_size_demo": 1024,            # batch size for the demonstrations.
                "is_uniform_sampling": False,       # if True, use the uniform sampling for the replay buffer.
                "is_demo_permanent": False,         # if True, the demonstrations are permanently stored in the replay buffer.
                "demo_sample_portion": None,        # portion of the demonstrations when sampling from the replay buffer.
                "decay_rate_demo_portion": 0.98,    # decay rate of the demo portion.
                
                "lr_pi": 1e-3,      # learning rate for the policy.
                "lr_q": 1e-3,       # learning rate for the Q-functions.
                "lr_disc": 1e-4,    # learning rate for the discriminator.
                "gamma": 0.99,      # discount factor.
                "polyak": 0.995,    # polyak averaging for the target networks.
                "alpha": 0.005,     # entropy regularization coefficient.
                "lamb": 10.0,       # adjustment parameter btw the demonstrations and the agent's experiences.

                "hidden_sizes": [2048, 2048, 1024, 512],    # hidden sizes for the policy and Q-functions.
                "hidden_sizes_disc": [1024, 512, 128, 64],  # hidden sizes for the discriminator.
                "lr_decay_pi": 1.0,                         # learning rate decay for the policy.
                "decay_every_pi": 12000,                    # period of the learning rate decay for the policy.
                "lr_decay_q": 1.0,                          # learning rate decay for the Q-functions.
                "decay_every_q": 12000,                     # period of the learning rate decay for the Q-functions.
                "lr_decay_disc": 1.0,                       # learning rate decay for the discriminator.
                "decay_every_disc": 12000,                  # period of the learning rate decay for the discriminator.
                
                "start_steps": 0,               # before start_steps, the action is random.
                "use_buffer_after": 0,          # after save_buffer_after, start to save replay_buffer. Until that, do not decay the demo portion.
                "update_after": 0,              # after update_after, the training starts.
                "update_every": 2000,           # update the network every update_every steps.
                "num_update": 300,              # number of updates at each update.
                "num_test_episodes": 1,         # number of test episodes.
                "max_ep_len": 20000,            # maximum length of the episode.
                "save_freq": 10                 # frequency of saving.
}

log_dir = "./log/"                                                  # directory for the log files.
file_dict = {                                                       # list of the files to be saved.
                "main_file":"./main_new.py",
                "env_file":"./env/ACEnv.py", 
                "controller_file":"./env/ac_controller.py",
                "route_manager_file":"./utils/route_manager.py",
                "util_file":"./utils/IS_ACUtil.py",
                "config_file":"./config_SAC_lambda.py",
                "algorithm_folder":"./algorithm"
            }

