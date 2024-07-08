# fed-gan
![image](https://github.com/daxx1/fed-gan/assets/92024670/cd21a13e-d68a-42c8-84a1-59df176e29ff)
python main.py --checkpoint_dir mnist_z_dim_50_topk_200_teacher_5000_sigma_2100_thresh_0.6_pt_30_d_step_2_stochastic_1e-5/ --topk 200 --signsgd --norandom_proj --shuffle  --teachers_batch 100 --batch_teachers 50 --dataset mnist --train --max_eps 3 --train --thresh 0.6 --sigma 2100 --nopretrain --z_dim 50 --nosave_epoch --epoch 300 --save_vote --d_step 2 --pretrain_teacher 10 --stochastic --max_grad 1e-5
