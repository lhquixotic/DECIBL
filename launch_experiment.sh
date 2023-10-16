#！/bin/sh
task_seq="1-2-3-4"
expand_thres=0.3
dynamic_expand=1

python launch.py --experiment_name experiment-2 \
                 --is_demo 0\
                 --start_task 3\
                 --task_seq $task_seq\
                 --dynamic_expand $dynamic_expand\
                 --expand_thres $expand_thres\
                 --batch_size 128\
                 --num_epochs 200\
                 --test_times 10\
                 --test_case_num 1000
