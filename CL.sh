
clear

set -x
set -e
set -u

experiment_name=$1

news_data_dir="/export/data2/chuber/2024/NewsData"
factorization_rank=4
dataset_factor=10000
memoryfilesname=0
min_utt_inc=20 # only learn adaptation weights if x more training utterances are available

if [ "$experiment_name" = "cb_only_memoryfiles0" ]; then
    min_utt_inc=99999999 # only context biasing
elif [ "$experiment_name" = "cl_0" ]; then
    :
else
    exit
fi

mkdir -p CL/$experiment_name/hypos
mkdir -p CL/$experiment_name/models
mkdir -p CL/$experiment_name/data

i=0
num_utt_adapt=0

adapter_model="./saves/model_baseline_adapt" # baseline whisper model adapted on cv and voxpopuli

while IFS= read -r talk; do
    segfile="$news_data_dir/segfiles/$talk.seg.aligned"
    memory_file="$news_data_dir/memory_files/$experiment_name/$talk.memory"
    hypofile="CL/$experiment_name/hypos/$talk.hyp"

    # Decode talk
    if [ ! -e "$hypofile" ]; then
        echo Decoding $segfile
        adapter_model_checkpoint=`ls -d $adapter_model/*`
        python decode_asr.py --model_path "./saves/model_newwords15/checkpoint-184000" --segfiles $segfile --use_memory --memory_file $memory_file --hypo_file $hypofile --load_adapter_model $adapter_model_checkpoint --batch_size 4 --num_beams 4
    fi

    # Generate pseudolabel data files
    echo Generating segfiles CL/$experiment_name/data/$talk.*
    num_utt=`python generate_pseudolabels.py $talk $i $news_data_dir $experiment_name $memoryfilesname | tail -n1`

    # Learn new factorization weights
    if (( num_utt > num_utt_adapt + min_utt_inc )); then
        if [ ! -e "CL/$experiment_name/models/model_$i" ]; then
            echo Learning new factorization weights CL/$experiment_name/models/model_$i
            python -u train.py --model_path CL/$experiment_name/models/model_$i \
                --load $adapter_model \
                --segfiles "data/cv.*.train.seg.aligned" "data/voxpopuli.EN.train.seg.aligned" "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.*.train.*.seg.aligned" "CL/$experiment_name/data/*.train.seg.aligned" \
                --dataset_factors 1 1 $dataset_factor $dataset_factor \
                --segfiles_dev "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.*.test.seg.aligned" "CL/$experiment_name/data/*.dev.seg.aligned" \
                --warmup_steps 0 --learning_rate 1e-5 \
                --log_steps 10 --use_early_stopping 2 \
                --eval_steps 10 \
                `#--gradient_checkpointing` \
                --factorization_rank $factorization_rank `#--factorization_only_decoder` \
                --batch_size 4 --gradient_accumulation_steps 8 
        fi

        adapter_model=CL/$experiment_name/models/model_$i
        num_utt_adapt=$num_utt
    fi

    i=$(($i+1))
done < "$news_data_dir/memory_files/order_$memoryfilesname.txt"

