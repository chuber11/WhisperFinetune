
set -x

GPU=${1:-2}
user=${2:-christian.huber@kit.edu}
#user="alexander.waibel@kit.edu"
server="30"
pipeline="ltpipeline-dev"
dataset_factor="10000"

mkdir -p data_to_process_$user
mkdir -p data_processed_$user

num_files=`ls -1 data_processed_$user | wc -l`

date_file="last_run_day_$user.txt"
if [[ -f "$date_file" ]]; then
	last_run_date=$(cat "$date_file")
else
    last_run_date=$(date +%Y-%m-%d)
    last_run_date=$(date -I -d "$last_run_date - 1 day")
fi

today=$(date +%Y-%m-%d)

# Iterate over dates not yet processed and collect data

current_date="$last_run_date"
while [[ "$current_date" < "$today" ]]; do
    current_date=$(date -I -d "$current_date + 1 day")
    echo Running date "$current_date"
    
    ssh mtasr@141.3.25.$server "cd /home/mtasr/LT2.0/$pipeline/extract_data && bash extract_data.sh $current_date $user"
    scp -r mtasr@141.3.25.$server:/home/mtasr/LT2.0/$pipeline/extract_data/data data_to_process_$user
    if ls data_to_process_$user/data/* 1> /dev/null 2>&1; then
        mv data_to_process_$user/data/* data_to_process_$user
    fi
    rm -r data_to_process_$user/data
done

# Segment, transcribe and filter data for pseudo labels containing new words

new_files=`ls -1 data_to_process_$user | wc -l`
if [ "$new_files" -gt 0 ]; then
    bash process_data.sh $user $GPU
fi

# Adapt model

num_files2=`ls -1 data_processed_$user | wc -l`
if [ "$num_files" != "$num_files2" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU
    cd ..
    bash train_factorization.sh -n CL_${user}_${current_date} None $dataset_factor $user
    cd -
    cp -r ../saves/model_CL_${user}_${current_date} /project/OML/chuber/2023/LT2.0/Whisper-memory/WhisperFinetune/saves
fi

echo $current_date > $date_file

