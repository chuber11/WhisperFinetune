
model=${1:-openai/whisper-large-v2}
adapter=${2:-None}
beam=${3:-4}
path=`echo $model | sed "s/\//_/g"`
write_at_end="" # "--no_write_at_end"

lang=EN
lang2=english

if [ "$adapter" = "None" ]; then
    python decode_asr.py --segfiles data_filtered_test/cv_filtered.$lang.test.seg.aligned --model_path $model --model_name saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350 --hypo_file hypos/hypo_${path}_cv_filtered_beam$beam.$lang.txt --num_beams $beam --language $lang2 $write_at_end --batch_size 6 --decode_only_first_part
else
    python decode_asr.py --segfiles data_filtered_test/cv_filtered.$lang.test.seg.aligned --model_path openai/whisper-large-v2 --load_adapter_model $model --model_name saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350 --hypo_file hypos/hypo_${path}_cv_filtered_beam$beam.$lang.txt --num_beams $beam --language $lang2 $write_at_end --batch_size 6 --decode_only_first_part
fi

lang=DE
lang2=german

if [ "$adapter" = "None" ]; then
    python decode_asr.py --segfiles data_filtered_test/cv_filtered.$lang.test.seg.aligned --model_path $model --model_name saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350 --hypo_file hypos/hypo_${path}_cv_filtered_beam$beam.$lang.txt --num_beams $beam --language $lang2 $write_at_end --batch_size 6 --decode_only_first_part
else
    python decode_asr.py --segfiles data_filtered_test/cv_filtered.$lang.test.seg.aligned --model_path openai/whisper-large-v2 --load_adapter_model $model --model_name saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350 --hypo_file hypos/hypo_${path}_cv_filtered_beam$beam.$lang.txt --num_beams $beam --language $lang2 $write_at_end --batch_size 6 --decode_only_first_part
fi

