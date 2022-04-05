# CROP: Certifying Robust Policies for Reinforcement Learning through Functional Smoothing

We propose CROP, the ﬁrst uniﬁed framework for certifying robust policies for RL against test-time evasion attacks on agent observations. In particular, we propose two robustness certiﬁcation criteria: *robustness of per-state actions* and *lower bound of cumulative rewards*. We then develop three novel methods (LoAct, GRe, LoRe) to achieve certification corresponding to the two certification criteria. More details can be found in our paper:

*Fan Wu, Linyi Li, Zijian Huang, Yevgeniy Vorobeychik, Ding Zhao*, and *Bo Li*, "CROP: Certifying Robust Policies for Reinforcement Learning through Functional Smoothing", [ICLR 2022](https://openreview.net/forum?id=HOjLHrlZhmx)

All experimental results are available at the website https://crop-leaderboard.github.io.

## Content of the repository

In our paper, we apply our **three** certiﬁcation algorithms (*CROP-LoAct, CROP-GRe*, and *CROP-LoRe*) to certify **nine** RL methods (*StdTrain, GaussAug, AdvTrain, SA-MDP (PGD,CVX), RadialRL, CARRL, NoisyNet*, and *GradDQN*) on two high-dimensional Atari games (*Pong* and *Freeway*), one low dimensional control environment (*CartPole*), and an autonomous driving environment (*Highway*). For all algorithms in all environments, we obtain certification on either per-state action stability or cumulative reward lower bound. 

In this repository, we provide the code for our CROP framework, built on top of the deep RL codebase [CleanRL](https://github.com/vwxyzjn/cleanrl). Basically, our repository contains both the code for 

1. training policies (via functionalities provided by CleanRL codebase), and
2. certifying the trained policies (via testing code and APIs in our CROP framework).

Below, we first present example commands for running the certifications (including LoAct, GRe, and LoAct), and then provide the usage of the easy-to-use plug-and-play APIs such that interested readers can directly integrate these certification APIs into their own testing code for their trained models.

## Example commands for certification

In this part, we present the example commands for obtaining certification corresponding to two certification criteria via three certification algorithms.

### CROP-LoAct

We first run the pre-processing step to obtain the output range of the Q-network, e.g.,

```bash
python cleanrl_estimate_q_range.py \
			--load-checkpoint <model_path> --dqn-type <model_type> \
			--m 10000 --sigma 0.01
```

Then, we update the configuration file `config_v_table.py` and run LoAct to obtain the certification for per-state action stability via local smoothing, e.g.,

```bash
python cleanrl_certify_r.py \
			--load-checkpoint <model_path> --dqn-type <model_type> \
			--m 10000 --sigma 0.01
```

The results are stored in files with the suffix `_certify-r-{i}.pt`.

### CROP-GRe

Example command to run GRe to obtain the certification for cumulative reward via global smoothing:

```bash
python cleanrl_run_global.py \
				--gym-id PongNoFrameskip-v4 --restrict-actions 4 \
				--load-checkpoint <model_path> --dqn-type <model_type> \
				--max-episodes 10000 --sigma 0.01
```

The results are stored in the file with the suffix `_global-reward.pt`.

### CROP-LoRe

Example command to run LoRe to obtain the certification for cumulative reward via adaptive search algorithm along with local smoothing: 

```bash
python cleanrl_certify_r.py \
				--gym-id PongNoFrameskip-v4 --restrict-actions 4 \
				--load-checkpoint <model_path> --dqn-type <model_type> \
				--m 10000 --sigma 0.01
```

The results are stored in the file with the suffix `_certify-map.pt`.

## Usage of APIs

### class LoAct

- **Filepath**: ``lo_act.py``
- **Class name**: ``LoAct``
- **Input variables**:

``log_func``: the function for logging information

``input_shape``: shape of the state observation

``model``: the model (Q-network)

``forward_func``: the model forward function of the given model (e.g., ``model.forward``). This function returns the Q-value.

``m``: number of samples for randomized smoothing

``sigma``: standard deviation of the smoothing Gaussian noise

``v_lo`` and ``v_hi``: the estimated output range of the Q-network. Details see Section 4.2 of the paper.

``conf_bound``: parameter for computing the confidence interval in the Hoeffding's inequality.

- **Functions**:

``__init__``: initialization

``init_record_list``: initialize the statistics to be saved

``forward``: 1) perform randomized smoothing; 2) compute the certification via Theorem 1 in Section 4.1

``save``: save the statistics and reset

* **How to incorporate the API**

1. *Model loading*: after loading the model as in the original testing, wrap the loaded model into ``LoAct``;
2. *Forwarding*: replace the original forwarding step via the model with the forwarding step via ``LoAct``;
3. *Statistics saving*: after finishing one episode, save the stored statistics and reset.

* **Example file for proper usage of the API**: ``cleanrl_certify_r.py``

### class GRe

- **Filepath**: ``g_re.py``
- **Class name**: ``GRe``

- **Input variables**:

``log_func``, ``input_shape``, ``model``, ``forward_func``, ``sigma``: same as described in the previous part for class LoAct

- **Functions**:

``__init__``: initialization

``forward``: perform global smoothing by adding one noise at each given time step

* **How to incorporate the API**

1. *Model loading*: after loading the model as in the original testing, wrap the loaded model into ``GRe``;
2. *Forwarding*: replace the original forwarding step via the model with the forwarding step via ``GRe``;
3. *Statistics saving*: after completing ``args.max_episodes`` number of trajectories via ``GRe`` forwarding and obtaining the cumulative rewards for these ``args.max_episodes`` randomized trajectories, save these reward values.

* **Example file for proper usage of the API**: ``cleanrl_run_global.py``

### class LoRe

- **Filepath**: ``lo_re.py``
- **Class name**: ``LoRe``
- **Input variables**:

``log_func``, ``input_shape``, ``model``, ``forward_func``, ``m``, ``sigma``, ``v_lo``, ``v_hi``, ``conf_bound``: same as described in the previous part for class LoAct

``max_frames_per_episode``: the trajectory/horizon length to evaluate for the reward certification, i.e., $H$

- **Main functions**:

``__init__``: initialization, including the preparation for priority queue and the memorization in search

``run``: the entire adaptive search algorithm, alternating between the *trajectory exploration and expansion* step and the *perturbation magnitude growth* step in the loop

``expand``: the *trajectory exploration and expansion* step

``take_action``: deciding the possible action set for each current step, via Theorem 4 in Section 5.2

``save``: saving ``certified_map`` which contains the list of mappings from perturbation magnitudes to the corresponding certified lower bounds to the corresponding file

* **How to incorporate the API**

1. *Model loading*: after loading the model as in the original testing, wrap the loaded model into ``LoAct``;
2. *Adaptive search*: directly call ``lo_re.run(env, obs)``, where ``obs`` is the fixed initial observation;

* **Example file for proper usage of the API**: ``cleanrl_tree.py``

* **Sidenote**: During the growth of the tree, we keep track of the nodes and edges corresponding to the states and transitions. The tree structure can be saved via ``save_tree`` at the end of the adaptive search, which facilitates the visualization of the search tree as well as the understanding of the certification procedure.

## Reference

```tex
@inproceedings{wu2022crop,
title={CROP: Certifying Robust Policies for Reinforcement Learning through Functional Smoothing},
author={Wu, Fan and Li, Linyi and Huang, Zijian and Vorobeychik, Yevgeniy and Zhao, Ding and Li, Bo},
booktitle={International Conference on Learning Representations},
year={2022}
}
```
