
clear

set -x
set -e
set -u

experiment_name=${1:-cb_only_memoryfiles0}

#news_data_dir="/export/data2/chuber/2024/NewsData"
news_data_dir="/export/data2/chuber/2024/CSData"
factorization_rank=8
memoryfilesname=0
memory_model="model_newwords18_3/checkpoint-128000"

if [ "$experiment_name" = "cb_only_memoryfiles0" ]; then
    min_utt_inc=99999999 # only context biasing
    memory_model="model_newwords18_2/checkpoint-94000"
elif [ "$experiment_name" = "cb_only_memoryfiles1" ]; then
    min_utt_inc=99999999 # only context biasing
    memory_model="model_newwords18_3/checkpoint-128000"
elif [ "$experiment_name" = "cb_only_memoryfiles2" ]; then
    memoryfilesname=1 # filtered words list
    min_utt_inc=99999999 # only context biasing
    memory_model="model_newwords18_2/checkpoint-94000"
elif [ "$experiment_name" = "cl_0" ]; then
    min_utt_inc=5 # only learn adaptation weights if x more training utterances are available
elif [ "$experiment_name" = "cl_1" ]; then
    memoryfilesname=1 # filtered words list
    min_utt_inc=4 # only learn adaptation weights if x more training utterances are available
    memory_model="model_newwords18_2/checkpoint-94000"
elif [[ "$experiment_name" == "qwen"* ]]; then
    min_utt_inc=99999999 # only context biasing
else
    exit
fi

mkdir -p CL/$experiment_name/hypos
mkdir -p CL/$experiment_name/models
mkdir -p CL/$experiment_name/data

i=0
num_utt_adapt=0

adapter_model_init="" #"./saves/model_baseline_adapt_yodas" # baseline whisper model adapted on yodas
adapter_model=$adapter_model_init

while IFS= read -r talk; do
    segfile="$news_data_dir/segfiles/$talk.seg.aligned"
    memory_file="$news_data_dir/memory_files/$memoryfilesname/$talk.memory"
    hypofile="CL/$experiment_name/hypos/$talk.hyp"

    # Decode talk
    if [ ! -e "$hypofile" ]; then
        echo Decoding $segfile
        if [[ -z $adapter_model ]]; then
            options=""
        else
            adapter_model_checkpoint=`ls -d $adapter_model/*`
            options="--load_adapter_model $adapter_model_checkpoint"
        fi
        python decode_asr.py --model_path "./saves/${memory_model}" --segfiles $segfile --use_memory --memory_file $memory_file --hypo_file $hypofile --batch_size 4 --num_beams 4 $options #--load_adapter_model $adapter_model_checkpoint
    fi

    # Generate pseudolabel data files
    echo Generating segfiles CL/$experiment_name/data/$talk.*
    #num_utt=`python generate_pseudolabels.py $talk $i $news_data_dir $experiment_name $memoryfilesname | tail -n1`
    num_utt=`python generate_pseudolabels.py $talk $i $news_data_dir $experiment_name $memoryfilesname`
    echo Found pseudolabels $num_utt

    # Learn new factorization weights
    if (( num_utt > num_utt_adapt + min_utt_inc )); then
        if [ ! -e "CL/$experiment_name/models/model_$i" ]; then
            # dataset_factor such that during 500 update the new data should be used twice?
            percent_new_data=50
            samples_yodas=1240000
            dataset_factor=$((($samples_yodas*100/$percent_new_data-$samples_yodas)/(465+$num_utt)))

            echo Learning new factorization weights CL/$experiment_name/models/model_$i
            python -u train.py --model_path CL/$experiment_name/models/model_$i \
                `#--load $adapter_model_init` \
                --load openai/whisper-large-v2 \
                --segfiles "data_yodas/EN.train.seg.aligned" "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.*.train.*.seg.aligned" "CL/$experiment_name/data/*.train.seg.aligned" \
                --dataset_factors 1 $dataset_factor $dataset_factor \
                --segfiles_dev "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.*.test.seg.aligned" "CL/$experiment_name/data/*.dev.seg.aligned" \
                --warmup_steps 0 --learning_rate 1e-4 \
                --log_steps 10 --use_early_stopping 2 \
                --eval_steps 20 \
                `#--gradient_checkpointing` \
                --factorization_rank $factorization_rank `#--factorization_only_decoder` \
                --batch_size 4 --gradient_accumulation_steps 4 --model_name saves/large_v2_files #--adapt_loaded_adapter
        fi

        adapter_model=CL/$experiment_name/models/model_$i
        num_utt_adapt=$num_utt
    fi

    i=$(($i+1))
done < "$news_data_dir/memory_files/order_$memoryfilesname.txt"

