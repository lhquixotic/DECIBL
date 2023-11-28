#！/bin/sh
task_seq="1-2-3-4-5"
train_method="DEM"
experiment_name="gae-train"

python launch.py --train_method $train_method\
                 --experiment_name $experiment_name\
                 --is_demo 0\
                 --train_start_task 0\
                 --test_start_task 0\
                 --task_seq $task_seq\
                 --batch_size 64\
                 --num_epochs 250\
                 --test_times 5\
                 --test_case_num 1000\
                 --datasets_split_num 2\
                 --init_new_expert 0\
                 --use_lrschd 1\
                 --lr_sh_rate 100\
                 --learning_rate 0.01\
                 --clip_grad 1.0\
                 --task_free 1\
                 --no_train 1\
                 --no_test 1\
                 --no_gae_train 0