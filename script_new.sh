# atari
# GradDQN, NoisyNet Training
CUDA_VISIBLE_DEVICES=0 python train_atari.py --dqn-type noisynet --gym-id PongNoFrameskip-v4 --restrict-actions 4 # NoisyNet
CUDA_VISIBLE_DEVICES=0 python train_atari.py --dqn-type graddqn --gym-id PongNoFrameskip-v4 --restrict-actions 4 # GradDQN

# RadialRL Training (in radial_rl-cleanrl/DQN folder)
CUDA_VISIBLE_DEVICES=0 python main.py --env PongNoFrameskip-v4 --robust --gpu-id 0 --num-act 4 # RadialRL

# Other model Training (in atari folder)
CUDA_VISIBLE_DEVICES=0 python dqn_atari_new.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 # StdTrain
CUDA_VISIBLE_DEVICES=0 python dqn_atari_new.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --adv-train # AdvTrain
CUDA_VISIBLE_DEVICES=0 python dqn_atari_new.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --gaussian-aug # GaussAug
CUDA_VISIBLE_DEVICES=0 python dqn_atari_new.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --robust --bound-solver cvx # SA-MDP (CVX)
CUDA_VISIBLE_DEVICES=0 python dqn_atari_new.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --robust --bound-solver pgd # SA-MDP (pgd)

# estimate q range
CUDA_VISIBLE_DEVICES=0 python cleanrl_estimate_q_range.py --load-checkpoint <model_path> --dqn-type <model_type> --m 1000 --sigma 0.001

# LoAct
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --load-checkpoint <model_path> --dqn-type <model_type> --m 10000 --sigma 0.001

# GRe 
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint <model_path> --dqn-type <model_type> --max-episodes 10000 --sigma 0.005 

# global attck 
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint <model_path> --dqn-type <model_type> --max-episodes 11 --sigma 0.001 --epsilon 0.2e-8

# LoRe
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint <model_path> --dqn-type <model_type> --m 10000 --sigma 0.001

# local attack 
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint <model_path> --dqn-type <model_type> --m 10000 --sigma 0.01 --epsilon 0.002


# CartPole 
# GradDQN, NoisyNet (in cartpole folder)
# Training 
CUDA_VISIBLE_DEVICES=0 python train_cartpole.py --dqn-type noisynet # NoisyNet
CUDA_VISIBLE_DEVICES=0 python train_cartpole.py --dqn-type graddqn # GradDQN

# estimate q range
CUDA_VISIBLE_DEVICES=0 python cleanrl_carpole_estimate_q_range.py --load-checkpoint <model_path> --dqn-type <model_type> --m 1000 --sigma 0.001

# LoAct
CUDA_VISIBLE_DEVICES=0 python cleanrl_cartpole_certify_r.py --load-checkpoint <model_path> --dqn-type <model_type> --m 10000 --sigma 0.001

# GRe
CUDA_VISIBLE_DEVICES=0 python cleanrl_cartpole_run_global.py --load-checkpoint <model_path> --dqn-type <model_type> --max-episodes 10000 --sigma 0.005 

# global attack
CUDA_VISIBLE_DEVICES=0 python cleanrl_cartpole_attack_global.py --load-checkpoint <model_path> --dqn-type <model_type> --max-episodes 11 --sigma 0.001 --epsilon 0.2e-8

# LoRe
CUDA_VISIBLE_DEVICES=0 python cleanrl_cartpole_certify_r.py --load-checkpoint <model_path> --dqn-type <model_type> --m 10000 --sigma 0.001

# local attack
CUDA_VISIBLE_DEVICES=0 python cleanrl_cartpole_attack.py --load-checkpoint <model_path> --dqn-type <model_type> --m 10000 --sigma 0.01 --epsilon 0.002

# RadialRL (in radial_rl_cartpole/DQN folder)
# Training 
CUDA_VISIBLE_DEVICES=0 python main_CartPole.py --robust --env CartPole-v0 

# estimate q range
CUDA_VISIBLE_DEVICES=0 estimate_q_range.py --env CartPole-v0 --gpu-id 0 --load-path <model_path> 

# LoAct
CUDA_VISIBLE_DEVICES=0 python certify_r_CartPole.py --env CartPole-v0 --gpu-id 0 --max-frames-per-episode 200 --robust --load-path "trained_models/CartPole-v0_2021-07-12_22_59_46_best.pt" --sigma 0.001 --m 10000 --num-episodes 10

# GRe (being checked)
CUDA_VISIBLE_DEVICES=0 python test_attack.py --env CartPole-v0  --robust --load-path "trained_models/CartPole-v0_2021-07-12_22_59_46_best.pt" --gpu-id 0 --m=1 --sigma=0.001 --max_frames_per_episode=500 --epsilon=0.00002

# global attack
CUDA_VISIBLE_DEVICES=0 python test_attack_global_CartPole.py --env CartPole-v0 --robust --load-path "trained_models/CartPole-v0_2021-07-12_22_59_46_best.pt" --gpu-id 0 --sigma 0.001 --epsilon 0.00002 --m 1 --max_frames_per_episode 200 --num_episodes 101

# LoRe
CUDA_VISIBLE_DEVICES=0 python test_tree_CartPole.py --env CartPole-v0 --robust --load-path "trained_models/CartPole-v0_2021-07-12_22_59_46_best.pt" --gpu-id 0 --m 10000 --sigma 0.001

# local attack
CUDA_VISIBLE_DEVICES=0 python test_attack_CartPole.py --env CartPole-v0 --robust --load-path "trained_models/CartPole-v0_2021-07-12_22_59_46_best.pt" --gpu-id 0 --sigma 0.001 --epsilon 0.00025 --m 10000 --max-frames-per-episode 10 --num_episodes 1

# Other model Training (in cartpole/CROP_code_CartPole folder, CARRL use the same model as StdTrain)
# Training 
CUDA_VISIBLE_DEVICES=0 python train.py --config config/CartPole_nat.json # StdTrain
CUDA_VISIBLE_DEVICES=0 python train.py --config config/CartPole_adv.json # AdvTrain
CUDA_VISIBLE_DEVICES=0 python train.py --config config/CartPole_aug.json # GaussAug
CUDA_VISIBLE_DEVICES=0 python train.py --config config/CartPole_cov.json # SA-MDP (CVX)
CUDA_VISIBLE_DEVICES=0 python train.py --config config/CartPole_pgd.json # SA-MDP (PGD)

# estimate q range
CUDA_VISIBLE_DEVICES=0 python estimate_q_range.py --config <config_file_path> 

# LoAct
CUDA_VISIBLE_DEVICES=0 python certify_r.py --config <config_file_path>

# GRe
CUDA_VISIBLE_DEVICES=0 python test_GS.py --config <config_file_path>

# global attack
CUDA_VISIBLE_DEVICES=0 python test_attack_global.py --config <config_file_path>

# LoRe
CUDA_VISIBLE_DEVICES=0 python test_tree.py --config <config_file_path>

# local attack
CUDA_VISIBLE_DEVICES=0 python test_attack.py --config <config_file_path>
 
# highway (use highway-fast-v0)
# GradDQN, NoisyNet Training
CUDA_VISIBLE_DEVICES=0 python train_highway.py --dqn-type noisynet --gym-id highway-fast-v0 # NoisyNet
CUDA_VISIBLE_DEVICES=0 python train_highway.py --dqn-type graddqn --gym-id highway-fast-v0 # GradDQN

# RadialRL Training (in radial_rl-cleanrl/DQN folder)
CUDA_VISIBLE_DEVICES=0 python main_highway.py --robust --env highway-fast-v0 --gpu-id 0 # RadialRL

# Other model Training (in atari folder)
CUDA_VISIBLE_DEVICES=0 python dqn_highway.py --gym-id highway-fast-v0 # StdTrain
CUDA_VISIBLE_DEVICES=0 python dqn_highway.py --gym-id highway-fast-v0 --adv-train # AdvTrain
CUDA_VISIBLE_DEVICES=0 python dqn_highway.py --gym-id highway-fast-v0 --gaussian-aug # GaussAug
CUDA_VISIBLE_DEVICES=0 python dqn_highway.py --gym-id highway-fast-v0 --robust --bound-solver cvx # SA-MDP (CVX)
CUDA_VISIBLE_DEVICES=0 python dqn_highway.py --gym-id highway-fast-v0 --robust --bound-solver pgd # SA-MDP (pgd)

# estimate q range
CUDA_VISIBLE_DEVICES=0 python cleanrl_estimate_q_range_highway.py --load-checkpoint <model_path> --dqn-type <model_type> --m 1000 --sigma 0.001

# LoAct
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r_highway.py --load-checkpoint <model_path> --dqn-type <model_type> --m 10000 --sigma 0.001

# GRe 
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global_highway.py --load-checkpoint <model_path> --dqn-type <model_type> --max-episodes 10000 --sigma 0.005 

# global attck 
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global_highway.py --load-checkpoint <model_path> --dqn-type <model_type> --max-episodes 11 --sigma 0.001 --epsilon 0.2e-8

# LoRe
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r_highway.py --load-checkpoint <model_path> --dqn-type <model_type> --m 10000 --sigma 0.001

# local attack 
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_highway.py --load-checkpoint <model_path> --dqn-type <model_type> --m 10000 --sigma 0.01 --epsilon 0.002