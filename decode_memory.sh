
model=${1:-saves/model_newwords8/checkpoint-199000}
distractors=${2:-0}
language=${3:-EN}
language2=${4:-english}
testset=${5:-cv}

memory=data_filtered_test/${testset}_memory.$language.test.allwords

model2=`echo $model | sed "s/\//_/g"`
memory2=`echo $memory | sed "s/\//_/g"`

python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1

