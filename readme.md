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


python ./train_cartpole_controller_diff.py  --scene_file ./data/scenes/carpole_on_water.json --lr 0.001 --method 1 --iters 100 --difffr --norm_grad --t 2 --test_only --control_interval 100 --bi 5 --model_path checkpoints/et1.0/difffr/g2/_normed/20260427-115330_lr_0.001/interval_5/no_t_max_loss_100/model_best_loss_1.420312e-04.pth --save_fluid_field
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


  python ./run_simulation_diff.py  --scene_file ./data/scenes/skip_stone_difffr_DFSPH.json --lr 0.2 --method 1 --reduce_lr --init_id 3 --difffr --train --iters 100 --gird_search

#### Fish Task

python ./train_controller_diff.py  --scene_file ./data/scenes/fish_on_water_new.json --lr 0.00001 --method 1 --iters 100 --difffr --norm_grad --custom_grad --t 5 --control_interval 50 --bi 15 --use_fluid --encoder_path fluid_encoder/models/20260506_003344/autoencoder_best_test_loss_2.182457e-03.pth --random_start --use_relative_state --time_in_state --model_path checkpoints/et3.0/difffr/g2_custom_const0.0/_normed/20260606-135936_lr_0.0001/interval_15/time_in_state_max_loss_100/model_best_loss_2.430687e-01.pth --test 



##### 向前游动任务：
checkpoints/et5.0/difffr/g2_custom_const0.0/_normed/20260608-031109_lr_5e-05/interval_15/time_in_state_max_loss_100/model_best_loss_1.077526e-03.pth


python ./train_controller_diff.py  --scene_file ./data/scenes/fish_on_water_new.json --lr 0.00001 --method 1 --iters 100 --difffr --norm_grad --custom_grad --t 5 --control_interval 50 --bi 15 --use_fluid --encoder_path fluid_encoder/models/20260506_003344/autoencoder_best_test_loss_2.182457e-03.pth --random_start --use_relative_state --time_in_state --model_path checkpoints/et5.0/difffr/g2_custom_const0.0/_normed/20260608-031109_lr_5e-05/interval_15/time_in_state_max_loss_100/model_best_loss_1.077526e-03.pth --add_gru  --use_v_loss --test


#### 保持平衡任务

python ./train_controller_diff.py  --scene_file ./data/scenes/fish_on_water_new.json --lr 0.00001 --method 1 --iters 100 --difffr --norm_grad --custom_grad --t 5 --control_interval 50 --bi 10 --use_fluid --encoder_path fluid_encoder/models/20260506_003344/autoencoder_best_test_loss_2.182457e-03.pth --use_relative_state --time_in_state --model_path checkpoints/et5.0/difffr_g2_custom_const0.0__normed/20260609-145908_lr_5e-05/interval_10/t_in_state_/model_best_loss_1.400100e-02.pth --test


## Reference

### reference resopitory

### Paper
