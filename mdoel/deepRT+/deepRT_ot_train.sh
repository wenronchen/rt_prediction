#!/bin/bash

function do1round
{
    # $1: job name
    # $2: seed
    # $3: job round number
    # $4: conv1 kernel
    # $5: conv2 kernel
    # $6: config.py
    echo "##### job: "${1}" | seed: "${2}" | round: "$3" #####"


    if [ ! -d ${1} ]; then
        mkdir ${1}
    fi
    if [ ! -d ${1}"/"${2} ]; then
        mkdir ${1}"/"${2}
    fi
    if [ ! -d ${1}"/"${2}"/"${3} ]; then
        mkdir ${1}"/"${2}"/"${3}
    fi
    
    
    echo "train_path = 'data/"${1}"_train_"${2}".txt'" > ${6} # 1
    echo "test_path = 'data/"${1}"_test_"${2}".txt' " >> ${6} # 2
    echo "result_path = 'work/"${1}"_pred_"${2}"_"${3}".txt'" >> ${6} # 3
    echo "log_path = 'work/"${1}"_"${2}"_"${3}".log'" >> ${6} # 4
    echo "save_prefix = 'work/"${1}"/"${2}"/"${3}"'" >> ${6} # 5
    echo "pretrain_path = ''" >> ${6} # 6
    echo "dict_path = ''" >> ${6} # 7

    echo "" >> ${6} 
    echo "conv1_kernel = "${4} >> ${6} # 8
    echo "conv2_kernel = "${5} >> ${6} # 9

    echo "min_rt = 0" >> ${6}
    echo "max_rt = 180" >> ${6}
    echo "time_scale = 60" >> ${6}
    echo "max_length = 200" >> ${6}

    cd ..
    python3 capsule_network_emb_cpu.py
    cd work
}


do1round $2 $1 3 8 8 "../config.py"

echo -e "done\n"
