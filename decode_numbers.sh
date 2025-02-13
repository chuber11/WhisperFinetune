
model=${1:-saves/model5/checkpoint-180}
adapter=${2:-None}
beam=${3:-4}
path=`echo $model | sed "s/\//_/g"`
write_at_end="" # "--no_write_at_end"

language=english
language2=EN

if [ "$adapter" = "None" ]; then
    python decode_asr.py --segfiles data_numbers/llm_augment.$language2.test.seg.aligned --model_path $model --model_name saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350 --hypo_file hypos/hypo_${path}_beam$beam.$language2.txt --num_beams $beam --language $language $write_at_end --batch_size 8
else
    python decode_asr.py --segfiles data_numbers/llm_augment.$language2.test.seg.aligned --model_path openai/whisper-large-v2 --load_adapter_model $model --model_name saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350 --hypo_file hypos/hypo_${path}_beam$beam.$language2.txt --num_beams $beam --language $language $write_at_end --batch_size 8
fi

language=german
language2=DE

if [ "$adapter" = "None" ]; then
    python decode_asr.py --segfiles data_numbers/llm_augment.$language2.test.seg.aligned --model_path $model --model_name saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350 --hypo_file hypos/hypo_${path}_beam$beam.$language2.txt --num_beams $beam --language $language $write_at_end --batch_size 8
else
    python decode_asr.py --segfiles data_numbers/llm_augment.$language2.test.seg.aligned --model_path openai/whisper-large-v2 --load_adapter_model $model --model_name saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350 --hypo_file hypos/hypo_${path}_beam$beam.$language2.txt --num_beams $beam --language $language $write_at_end --batch_size 8
fi

