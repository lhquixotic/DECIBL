#！/bin/sh
task_seq="1-2-3"
expand_thres=0.3
dynamic_expand=1

python launch.py --experiment_name demo-1 \
                 --is_demo 1\
                 --start_task 0\
                 --task_seq $task_seq\
                 --dynamic_expand $dynamic_expand\
                 --expand_thres $expand_thres\
                 --batch_size 5\
                 --num_epochs 5\
                 --test_times 5\
                 --test_case_num 10
                 