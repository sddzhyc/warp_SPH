A SPH-based Fluid-Rigid Coupling Simulator implemented by Warp.

## Implemented:

- WCSPH Fluid simulation

- two-way fluid-rigid coupling (use Ankici. 2012) 

## TODO

- differentable WCSPH with fluid-rigid

## Installation

```
conda env create -n warp_SPH python=3.9
conda activate warp_SPH
pip install -r requirements.txt
```

## Run
```
python .\run_simulation.py --scene_file .\data\scenes\warp_SPH_test.json

python .\run_simulation.py --scene_file data\scenes\rigid-fluid-demo.json


# python .\run_simulation.py --scene_file .\data\scenes\warp_SPH_test.json

```
### differentable simulation test
```
python .\run_simulation_diff.py --scene_file data\scenes\diff-demo.json --train

python .\run_simulation_diff.py --scene_file data\scenes\diff-demo.json --train --ply_path .\ply_states\particle_object_000030.ply
```

python .\run_simulation_diff.py --scene_file data\scenes\diff-demo2-1.json --train --iters 700 --lr 0.005

#### stone skipping

 python .\run_simulation_diff.py --ply_path .\skip_stones\  --scene_file .\data\scenes\skip_stone_difffr.json --train --iters 400 --lr 0.05 --num_timesteps 1400

  python .\run_simulation_diff.py --ply_path .\skip_stones\  --scene_file .\data\scenes\skip_stone_difffr.json --train --iters 400 --lr 0.05 --num_timesteps 1400

## Reference

### reference resopitory

### Paper
