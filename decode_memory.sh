
model=${1:-saves/model_newwords18_3/checkpoint-128000}
distractors=${2:-0}
language=${3:-EN}
language2=${4:-english}
testset=${5:-earnings}
addscore=${6:-0}
adapter=${7:-""}

memory=data_filtered_test/${testset}_memory.$language.test.allwords

model2=`echo $model | sed "s/\//_/g"`
memory2=`echo $memory | sed "s/\//_/g"`
adapter2=`echo $adapter | sed "s/\//_/g"`

python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory_diss/$model2$adapter2.$language.$memory2.$distractors.$addscore.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore #--load_adapter_model $adapter

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2$adapter2.$language.$memory2.$distractors.$addscore.logminf.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore #--load_adapter_model $adapter

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.boostall.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_other.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 #--force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_other_logminf.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 #--force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_text_other.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 #--force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.oracle_logminf.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.oracle_text.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_other_plus_text.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 #--force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_other_plus_text_logminf.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 #--force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_unsupervised.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore

# OLD

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_filtered.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_filtered_other.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_text_filtered.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore

#python decode_asr.py --segfiles data_filtered_test/${testset}_memory.$language.test.seg.aligned --model_path $model --model_name $model --use_memory --memory_file $memory --language $language2 --hypo_file hypos_memory/$model2.$language.$memory2.$distractors.$addscore.replacements_text_filtered_other.hyp --no_write_at_end --memory_num_distractors $distractors --batch_size 1 --force_exact_memory $addscore

