# AC-Racer

## Description
Autonomous car racing in Assetto Corsa (AC)


## Overview
Two main folders:

1. `apps` : Applications running on AC.
    - `lua` : Lua scripts. (Enable CSP in Content Manager to activate Lua apps.)
        - `teleport_vehicles` : Function for random vehicle spawn initialization.
        - `track_extractor` : Function for extracting track data.
    - `python` : Python scripts.
        - `DataParser` : Function for parsing data from AC into JSON files in real time.
            - `ac_api` : Parsing scripts using Python and Shared Memory APIs.
            - `data_bucket` : Directory for saving parsed data.
            - `third_party` : Third-party libraries.
            - `tracks` : Track data.
        - `IS_AddShortcutKey` : Function for adding shortcut keys.

2. `gym` : Gym for Reinforcement Learning.
    - `algorithm` : RL algorithms.
    - `env` : Virtual environment and controller.
    - `log` : Directory for saving data.
    - `utils` : Various utility scripts.


## Requirements
- Windows > 7
- Download Assetto Corsa from STEAM
- pip > 19.0 (If you have an older version, run:  ```pip install --upgrade pip``` )
- Download AutoHotkey == v1.1.37.01 from: [GitHub Releases](https://github.com/AutoHotkey/AutoHotkey/releases)
- Download AC ContentManager from: [AssettoCorsa.club](https://assettocorsa.club/content-manager.html)
- Install mpi4py. See: [Stack Overflow Guide](https://stackoverflow.com/questions/54615336/how-to-install-mpi4py-on-windows-10-with-msmpi)

Below is our experiment setup. (Recommanded)
- torch == 2.2.1
- CUDA 12.1
- CPU : 12th Gen Intel(R) Core(TM) i9-12900K
- GPU : NVIDIA GeForce RTX 3080 Ti


## Getting started
1. Drag & drop `comfy_map.zip` into content manager. Source: [RaceDepartment](https://www.racedepartment.com/downloads/comfy-map.52623/).
2. Create symbolic link for `apps/python/Dataparser` in your AC directory. For example, run the following commands in cmd run as administrator
    ```bash
    mklink /d "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\python\DataParser" "C:\Users\User\(your git directory)\AC-Racer\apps\python\DataParser"
    mklink /d "C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\lua\teleport_vehicle" "C:\Users\User\(your git directory)\AC-Racer\apps\lua\teleport_vehicle"
    ```
3. Repeat the same procedure for `apps/python/IS_AddShortcutKey`.
4. Start AC, go to the General Settings tab, and activate `DataParser` and `IS_AddShortcutKey` apps.
5. Now, real-time data will be saved in `apps/data_bucket/exp_*/exp.json`.
6. Install necessary libraries :
    ```bash
    cd gym
    pip install -r requirements.txt
    ```
7. Install PyTorch. We use :
    ```bash
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```
8. Done! Start RL training :
    ```bash
    python main.py
    ```
