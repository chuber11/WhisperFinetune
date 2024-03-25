
model=${1:-openai/whisper-large-v2}
adapter=${2:-None}
beam=${3:-4}
path=`echo $model | sed "s/\//_/g"`
write_at_end="" # "--no_write_at_end"

lang=EN
lang2=english

if [ "$adapter" = "None" ]; then
    python decode_asr.py --segfiles data_filtered_test/cv_filtered.$lang.test.seg.aligned --model_path $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_cv_filtered_beam$beam.$lang.txt --num_beams $beam --language $lang2 $write_at_end --batch_size 6 --decode_only_first_part
else
    python decode_asr.py --segfiles data_filtered_test/cv_filtered.$lang.test.seg.aligned --model_path openai/whisper-large-v2 --load_adapter_model $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_cv_filtered_beam$beam.$lang.txt --num_beams $beam --language $lang2 $write_at_end --batch_size 6 --decode_only_first_part
fi

lang=DE
lang2=german

if [ "$adapter" = "None" ]; then
    python decode_asr.py --segfiles data_filtered_test/cv_filtered.$lang.test.seg.aligned --model_path $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_cv_filtered_beam$beam.$lang.txt --num_beams $beam --language $lang2 $write_at_end --batch_size 6 --decode_only_first_part
else
    python decode_asr.py --segfiles data_filtered_test/cv_filtered.$lang.test.seg.aligned --model_path openai/whisper-large-v2 --load_adapter_model $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_cv_filtered_beam$beam.$lang.txt --num_beams $beam --language $lang2 $write_at_end --batch_size 6 --decode_only_first_part
fi

