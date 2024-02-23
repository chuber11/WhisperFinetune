
clear

set -x

experiment_name="${1:-cl1}"
factorization_rank="${2:-16}"
news_data_dir="/export/data1/chuber/2024/NewsData"

mkdir -p CL/$experiment_name/hypos
mkdir -p CL/$experiment_name/models
mkdir -p CL/$experiment_name/data

i=0

adapter_model="./saves/model_cl_start"

while IFS= read -r line; do
    talk=`echo $line | cut -d" " -f1`
    talk="${talk:8:${#talk}-20}"

    segfile="$news_data_dir/segfiles/$talk.seg.aligned"
    memory_file="$news_data_dir/memory_files/$talk.memory"
    hypofile="CL/$experiment_name/hypos/$talk.hyp"

    # Decode talk
    if [ ! -e "$hypofile" ]; then
        echo Decoding $segfile
        python decode_asr.py --segfiles $segfile --use_memory --memory_file $memory_file --hypo_file $hypofile --load_adapter_model $adapter_model --batch_size 2
    fi

    # Generate pseudolabel data files
    if [ ! -e "CL/$experiment_name/data/$talk.train.seg.aligned" ]; then
        echo Generating segfiles CL/$experiment_name/data/$talk.*
        python generate_pseudolabels.py $talk $i $news_data_dir CL/$experiment_name
    fi

    # Learn new factorization weights
    if [ -s "CL/$experiment_name/data/$talk.train.seg.aligned" ] || [ -s "CL/$experiment_name/data/$talk.dev.seg.aligned" ]; then
        if [ ! -e "CL/$experiment_name/models/model_$i" ]; then
            echo Learning new factorization weights CL/$experiment_name/models/model_$i
            python -u train.py --model_path CL/$experiment_name/models/model_$i \
                --load $adapter_model \
                --segfiles "data/cv.*.train.seg.aligned" "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.dev.train.*.seg.aligned" "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.test.train.*.seg.aligned" "CL/$experiment_name/data/*.train.seg.aligned" \
                --dataset_factors 1 $dataset_factor $dataset_factor $dataset_factor \
                --segfiles_dev "/project/OML/chuber/2023/data/earnings_nw_dataset/aligned_21/nw.dev.test.seg.aligned" "CL/$experiment_name/data/*.dev.seg.aligned" \
                --warmup_steps 0 --learning_rate 1e-5 \
                --log_steps 10 \
                --eval_steps 10 \
                `#--gradient_checkpointing` \
                --factorization_rank $factorization_rank `#--factorization_only_decoder` \
                --batch_size 2 --gradient_accumulation_steps 16 
        fi

        adapter_model=CL/$experiment_name/models/model_$i
    fi

    i=$(($i+1))

    break
done < "$news_data_dir/order.txt"

