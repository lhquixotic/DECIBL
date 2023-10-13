#！/bin/bash

declare -A dictionary
dictionary[1]="MA"
dictionary[2]="FT"
dictionary[3]="ZS"
dictionary[4]="EP"
dictionary[5]="SR"

echo "Test some dataset using different models"

dataset='1-MA'

for i in 1
do  
    # loaded best model trained on precedent tasks
    load_tag='2-FT'
    # for ((j=1;j<$i;j++))
    #     do  
    #         if [ $j -eq $((i-1)) ];then
    #             load_tag+=${dictionary[$j]}
    #         else
    #             load_tag+=${dictionary[$j]}'-'
    #         fi
    #     done
    echo 'model load tag is '$load_tag
    # args for python scripts
    cur_task=$i # test task dataset
    echo 'load dataset is '$dataset
    is_demo=0
    train_method='psstgcnn'
    model_name='social-stgcnn'

    python3 test.py --train_method $train_method --cur_task $cur_task --dataset $dataset --is_demo $is_demo --load_tag $load_tag --model_name $model_name
done


