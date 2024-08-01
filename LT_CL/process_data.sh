
user=${1:-admin@example.com}
GPU=${2:-2}

python transcribe.py $user

export CUDA_VISIBLE_DEVICES=$GPU
python generate_segments.py $user

python transcribe_memory.py $user

for s in `ls data_to_process_$user/*.train.seg.aligned`
do
    id=`echo $s | cut -d"/" -f2- | rev | cut -d"." -f4- | rev`
    mv data_to_process_$user/$id.mp3 data_processed_$user
    mv data_to_process_$user/$id.{train,dev}.seg.aligned data_processed_$user
    mv data_to_process_$user/$id.{train,dev}.hypo data_processed_$user
    mv data_to_process_$user/$id.{train,dev}.memory data_processed_$user
done
rm -rf data_to_process_$user

