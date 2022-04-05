
# training

python train.py --dqn-type <nature/noisynet/graddqn> --buffer-size 100000

# testing noisynet

python test.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet

# testing nature dqn

python test.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature



##################################################################
#################################               0731 - estimate q range
##################################################################


CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 1000 --sigma 0.001
CUDA_VISIBLE_DEVICES=3 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 1000 --sigma 0.005
CUDA_VISIBLE_DEVICES=0 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 1000 --sigma 0.01
CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 1000 --sigma 0.03
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 1000 --sigma 0.05
CUDA_VISIBLE_DEVICES=3 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 1000 --sigma 0.1




CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 1000 --sigma 0.001
CUDA_VISIBLE_DEVICES=3 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 1000 --sigma 0.005
CUDA_VISIBLE_DEVICES=0 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 1000 --sigma 0.01
CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 1000 --sigma 0.03
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 1000 --sigma 0.05
CUDA_VISIBLE_DEVICES=3 python cleanrl_estimate_q_range.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 1000 --sigma 0.1



##################################################################
#################################               0731 - certify R
##################################################################



CUDA_VISIBLE_DEVICES=2 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.001;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.01;\

CUDA_VISIBLE_DEVICES=3 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.03;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.1;\


CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.001;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.01;\

CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.03;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.1;\




##################################################################
#################################               0731 - tree
##################################################################



CUDA_VISIBLE_DEVICES=2 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.001 --conf-bound 0.1 ;\

CUDA_VISIBLE_DEVICES=2 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.005 --conf-bound 0.1 ;\

CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.01 --conf-bound 0.1 ;\





CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.03 --conf-bound 0.1 ;\

CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.05 --conf-bound 0.1 ;\

CUDA_VISIBLE_DEVICES=2 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.1 --conf-bound 0.1 ;\



CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.03 --conf-bound 0.2 ;\

CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.05 --conf-bound 0.2 ;\

CUDA_VISIBLE_DEVICES=2 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --m 10000 --sigma 0.1 --conf-bound 0.2 ;\






CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.001 --conf-bound 0.1 ;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.005 --conf-bound 0.1 ;\

CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.01 --conf-bound 0.1 ;\





CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.03 --conf-bound 0.1 ;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.05 --conf-bound 0.1 ;\

CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.1 --conf-bound 0.1 ;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.03 --conf-bound 0.2 ;\

CUDA_VISIBLE_DEVICES=2 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.05 --conf-bound 0.2 ;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_tree.py --load-checkpoint BoxingNoFrameskip-v4_noisynet/frame_3900000_reward90.2.pth --dqn-type noisynet --m 10000 --sigma 0.1 --conf-bound 0.2 ;\




##################################################################
#################################               0802 - train atari [cleanrl] (noisynet)
##################################################################


CUDA_VISIBLE_DEVICES=0 python train_atari.py --dqn-type noisynet --gym-id FreewayNoFrameskip-v4 --restrict-actions 3
CUDA_VISIBLE_DEVICES=3 python train_atari.py --dqn-type noisynet --gym-id PongNoFrameskip-v4 --restrict-actions 4 --buffer-size 80000


FreewayNoFrameskip-v4_noisynet/frame_200000_reward22.1.pth
PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth

CUDA_VISIBLE_DEVICES=3 python train_atari.py --dqn-type graddqn --gym-id FreewayNoFrameskip-v4 --restrict-actions 3


CUDA_VISIBLE_DEVICES=2 python train_atari.py --dqn-type graddqn --gym-id PongNoFrameskip-v4 --restrict-actions 4




##################################################################
#################################               0802 - estimate q range -- noisynet -- Freeway
##################################################################

# tab2.4, ai2 [done]

CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 1000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 1000 --sigma 0.05

CUDA_VISIBLE_DEVICES=3 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 1000 --sigma 0.1;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 1000 --sigma 0.5

CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 1000 --sigma 0.75;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 1000 --sigma 1.0



##################################################################
#################################               0803 - global smoothing -- noisynet -- Freeway
##################################################################

# tab1.4, ai3 [done]

CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.1;\

CUDA_VISIBLE_DEVICES=2 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.5;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.75;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 10000 --sigma 1.0;\





##################################################################
#################################               0802 - estimate q range -- noisynet -- Pong
##################################################################

# tab4.0, aws1 [done]

CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 1000 --sigma 0.001;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 1000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 1000 --sigma 0.01;\

CUDA_VISIBLE_DEVICES=3 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 1000 --sigma 0.03;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 1000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 1000 --sigma 0.1



##################################################################
#################################               0803 - global smoothing -- noisynet -- Pong
##################################################################

# tab4.0, aws1 [done]

CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.001;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.005;\

CUDA_VISIBLE_DEVICES=2 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.01;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.03;\

CUDA_VISIBLE_DEVICES=3 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 10000 --sigma 0.1;\




##################################################################
#################################               0803 - certify R -- noisynet -- Freeway
##################################################################

# # tab2.4, ai2 [done]
# tab4.5.1, aws1 [running]

CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 10000 --sigma 0.1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 10000 --sigma 0.5;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 10000 --sigma 0.75;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --m 10000 --sigma 1.0;\



##################################################################
#################################               0803 - certify R -- noisynet -- Pong
##################################################################

# # tab1.4, ai3 [done]
# tab4.5.2, aws1 [running]

CUDA_VISIBLE_DEVICES=3 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 10000 --sigma 0.001;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 10000 --sigma 0.01;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 10000 --sigma 0.03;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --m 10000 --sigma 0.1;\



##################################################################
#################################               0803 - attempt test
##################################################################


CUDA_VISIBLE_DEVICES=3 python test.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_1300000_reward20.9.pth --dqn-type graddqn

CUDA_VISIBLE_DEVICES=3 python test_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet

CUDA_VISIBLE_DEVICES=3 python test_attack.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature

CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --load-checkpoint BoxingNoFrameskip-v4_nature/frame_4400000_reward92.7.pth --dqn-type nature --max-episodes 101 --sigma 0.005


##################################################################
#################################               0803 - global attack -- noisynet -- Pong
##################################################################



CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 11 --sigma 0.001 --epsilon 0.2e-8;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 11 --sigma 0.001

CUDA_VISIBLE_DEVICES=3 python test.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-frames-per-episode 500

# tab4.4.1, aws1 [done]
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.001 --epsilon 0.2e-4;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.001 --epsilon 0.6e-4;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.001 --epsilon 1.0e-4;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.001 --epsilon 1.4e-4;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.005 --epsilon 1e-4;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.005 --epsilon 3e-4;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.005 --epsilon 5e-4;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.005 --epsilon 7e-4;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.01 --epsilon 0.2e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.01 --epsilon 0.6e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.01 --epsilon 1.0e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.01 --epsilon 1.4e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.03 --epsilon 1e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.03 --epsilon 2e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.03 --epsilon 3e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.03 --epsilon 4e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.05 --epsilon 1e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.05 --epsilon 3e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.05 --epsilon 5e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.05 --epsilon 7e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.1 --epsilon 0.2e-2;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.1 --epsilon 0.6e-2;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.1 --epsilon 1.0e-2;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_noisynet/frame_4000000_reward21.0.pth --dqn-type noisynet --max-episodes 101 --sigma 0.1 --epsilon 1.4e-2;\




##################################################################
#################################               0803 - global attack -- noisynet -- Freeway
##################################################################

# tab4.4.2, aws1 [done]

CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.005 --epsilon 1e-4;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.005 --epsilon 3e-4;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.005 --epsilon 5e-4;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.005 --epsilon 7e-4;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.05 --epsilon 1e-3;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.05 --epsilon 3e-3;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.05 --epsilon 5e-3;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.05 --epsilon 7e-3;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.1 --epsilon 0.2e-2;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.1 --epsilon 0.6e-2;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.1 --epsilon 1.0e-2;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.1 --epsilon 1.4e-2;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.5 --epsilon 1e-2;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.5 --epsilon 3e-2;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.5 --epsilon 5e-2;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.5 --epsilon 7e-2;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.75 --epsilon 0.2e-1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.75 --epsilon 0.4e-1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.75 --epsilon 0.6e-1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.75 --epsilon 0.8e-1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 0.75 --epsilon 1.0e-1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 1 --epsilon 0.2e-1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 1 --epsilon 0.6e-1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 1 --epsilon 1.0e-1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_noisynet/frame_4000000_reward21.2.pth --dqn-type noisynet --max-episodes 101 --sigma 1 --epsilon 1.4e-1;\




##################################################################
#################################               0806 - estimate q range -- graddqn -- Pong
##################################################################


# tab1.4.1, ai3 [done]
CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 1000 --sigma 0.001;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 1000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 1000 --sigma 0.01;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 1000 --sigma 0.03;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 1000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_estimate_q_range.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 1000 --sigma 0.1



##################################################################
#################################               0806 - estimate q range -- graddqn -- Freeway
##################################################################

# tab1.4.2, ai3 [done]

CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 1000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 1000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 1000 --sigma 0.1;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 1000 --sigma 0.5;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 1000 --sigma 0.75;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_estimate_q_range.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 1000 --sigma 1.0





##################################################################
#################################               0806 - global smoothing -- graddqn -- Freeway
##################################################################

# tab2.4.1, ai2 [done]
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.1;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.5;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.75;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_run_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 10000 --sigma 1.0;\



##################################################################
#################################               0806 - global smoothing -- graddqn -- Pong
##################################################################


# tab2.4.2, ai2 [done]
CUDA_VISIBLE_DEVICES=1 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.001;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.01;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.03;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_run_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 10000 --sigma 0.1;\




FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth
PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth



##################################################################
#################################               0806 - global attack -- graddqn -- Pong
##################################################################


# tab4.4.3, aws1 [done]
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.001 --epsilon 0.2e-4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.001 --epsilon 0.6e-4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.001 --epsilon 1.0e-4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.001 --epsilon 1.4e-4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.005 --epsilon 1e-4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.005 --epsilon 3e-4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.005 --epsilon 5e-4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.005 --epsilon 7e-4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.01 --epsilon 0.2e-3;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.01 --epsilon 0.6e-3;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.01 --epsilon 1.0e-3;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.01 --epsilon 1.4e-3;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.03 --epsilon 1e-3;\


# tab4.4.3, aws1 [done]
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.03 --epsilon 2e-3;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.03 --epsilon 3e-3;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.03 --epsilon 4e-3;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.05 --epsilon 1e-3;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.05 --epsilon 3e-3;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.05 --epsilon 5e-3;\

# tab2.5.3, ai2 [done]
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.05 --epsilon 7e-3;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.1 --epsilon 0.2e-2;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.1 --epsilon 0.6e-2;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.1 --epsilon 1.0e-2;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack_global.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --max-episodes 101 --sigma 0.1 --epsilon 1.4e-2;\




##################################################################
#################################               0806 - global attack -- graddqn -- Freeway
##################################################################

# tab1.5.2, ai3 [done]

CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.005 --epsilon 1e-4;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.005 --epsilon 3e-4;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.005 --epsilon 5e-4;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.005 --epsilon 7e-4;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.05 --epsilon 1e-3;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.05 --epsilon 3e-3;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.05 --epsilon 5e-3;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.05 --epsilon 7e-3;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.1 --epsilon 0.2e-2;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.1 --epsilon 0.6e-2;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.1 --epsilon 1.0e-2;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.1 --epsilon 1.4e-2;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.5 --epsilon 1e-2;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.5 --epsilon 3e-2;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.5 --epsilon 5e-2;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.5 --epsilon 7e-2;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.75 --epsilon 0.2e-1;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.75 --epsilon 0.4e-1;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.75 --epsilon 0.6e-1;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.75 --epsilon 0.8e-1;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 0.75 --epsilon 1.0e-1;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 1 --epsilon 0.2e-1;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 1 --epsilon 0.6e-1;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 1 --epsilon 1.0e-1;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack_global.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --max-episodes 101 --sigma 1 --epsilon 1.4e-1;\





##################################################################
#################################               0806 - certify R -- graddqn -- Pong
##################################################################

# TODO: rerun
PongNoFrameskip-v4_graddqn_m-10000_sigma-0.001_rerun:

# # tab1.4.1, ai3 [done]
# tab4.5.3, aws1 [running]
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.001;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.01;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.03;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_certify_r.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.1;\




##################################################################
#################################               0806 - local attack -- graddqn -- Pong
##################################################################


CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.001 --epsilon 0.0002;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.001 --epsilon 0.0006;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.001 --epsilon 0.0010;\
# tab2.5.1, ai2 [running]
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.001 --epsilon 0.0014;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.005 --epsilon 0.001;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.005 --epsilon 0.003;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.005 --epsilon 0.005;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.005 --epsilon 0.007;\

# tab2.5.2, ai2 [running]
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.01 --epsilon 0.002;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.01 --epsilon 0.004;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.01 --epsilon 0.006;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.01 --epsilon 0.008;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.01 --epsilon 0.010;\

# tab1.5.4, ai3 [running]
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.03 --epsilon 0.005;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.03 --epsilon 0.010;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.03 --epsilon 0.015;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.03 --epsilon 0.020;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.03 --epsilon 0.025;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.05 --epsilon 0.01;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.05 --epsilon 0.02;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.05 --epsilon 0.03;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.05 --epsilon 0.04;\

# tab4.4.1, aws1 [running]
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.1 --epsilon 0.02;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.1 --epsilon 0.04;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.1 --epsilon 0.06;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.1 --epsilon 0.08;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_attack.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.1 --epsilon 0.10;\



##################################################################
#################################               0806 - local attack -- graddqn -- Freeway
##################################################################

# tab1.5.1, ai3 [running]
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.005 --epsilon 0.002;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.005 --epsilon 0.004;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.005 --epsilon 0.006;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.005 --epsilon 0.008;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.005 --epsilon 0.010;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.05 --epsilon 0.02;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.05 --epsilon 0.04;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.05 --epsilon 0.06;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.05 --epsilon 0.08;\
CUDA_VISIBLE_DEVICES=2 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.05 --epsilon 0.10;\
# tab4.4.2, aws1 [running]
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.1 --epsilon 0.05;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.1 --epsilon 0.10;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.1 --epsilon 0.15;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.1 --epsilon 0.20;\

# tab1.5.3, ai3 [running]
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.5 --epsilon 0.2;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.5 --epsilon 0.4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.5 --epsilon 0.6;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.5 --epsilon 0.8;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.5 --epsilon 1.0;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.75 --epsilon 0.4;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.75 --epsilon 0.8;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.75 --epsilon 1.2;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.75 --epsilon 1.6;\
# tab4.4.3, aws1 [running]
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 1 --epsilon 0.25;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 1 --epsilon 0.75;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 1 --epsilon 1.25;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_attack.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 1 --epsilon 1.75;\


##################################################################
#################################               0731 - tree - graddqn - Freeway
##################################################################

# tab4.0.x, aws1 [done]
CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=0 python cleanrl_tree.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.1;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_tree.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.5;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_tree.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.75;\
CUDA_VISIBLE_DEVICES=3 python cleanrl_tree.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 1.0;\


##################################################################
#################################               0731 - tree - graddqn - Pong
##################################################################

# tab4.0.2, aws1 [done]
CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.001 --no-memo;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.005 --no-memo;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.01 --no-memo;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.03 --no-memo;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.05 --no-memo;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_tree.py --gym-id PongNoFrameskip-v4 --restrict-actions 4 --load-checkpoint PongNoFrameskip-v4_graddqn/frame_2000000_reward20.4.pth --dqn-type graddqn --m 10000 --sigma 0.10 --no-memo;\




##################################################################
#################################               0806 - certify R -- graddqn -- Freeway
##################################################################



# tab2.4.1, ai2 [running]
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.005;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.05;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.1;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.5;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 0.75;\
CUDA_VISIBLE_DEVICES=1 python cleanrl_certify_r.py --gym-id FreewayNoFrameskip-v4 --restrict-actions 3 --load-checkpoint FreewayNoFrameskip-v4_graddqn/frame_2000000_reward21.5.pth --dqn-type graddqn --m 10000 --sigma 1.0





# TODO: 


1. check GradDQN on Pong [empirical] -- median GradDQN Pong flat

2. compare Freeway median results (new vs in paper) -- 

update (Freeway, global smoothing, sigma=0.5, GaussAug)


3. rerun
PongNoFrameskip-v4_graddqn_m-10000_sigma-0.001_rerun:





========================================

CartPole: done

Pong Freeway:
LoAct: R done, reward running
GRe: certified done, empirical running
LoRe: certified done, empirical running


========================================


