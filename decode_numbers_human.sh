
model=${1:-saves/model5/checkpoint-180}
adapter=${2:-None}
beam=${3:-4}
path=`echo $model | sed "s/\//_/g"`
write_at_end="" # "--no_write_at_end"

language=english
language2=EN

if [ "$adapter" = "None" ]; then
    python decode_asr.py --segfiles data_numbers_human/llm_augment_human.$language2.test.seg.aligned --model_path $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_beam$beam.$language2.human.txt --num_beams $beam --language $language $write_at_end --batch_size 8
else
    python decode_asr.py --segfiles data_numbers_human/llm_augment_human.$language2.test.seg.aligned --model_path openai/whisper-large-v2 --load_adapter_model $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_beam$beam.$language2.human.txt --num_beams $beam --language $language $write_at_end --batch_size 8
fi

language=german
language2=DE

if [ "$adapter" = "None" ]; then
    python decode_asr.py --segfiles data_numbers_human/llm_augment_human.$language2.test.seg.aligned --model_path $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_beam$beam.$language2.human.txt --num_beams $beam --language $language $write_at_end --batch_size 8
else
    python decode_asr.py --segfiles data_numbers_human/llm_augment_human.$language2.test.seg.aligned --model_path openai/whisper-large-v2 --load_adapter_model $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_beam$beam.$language2.human.txt --num_beams $beam --language $language $write_at_end --batch_size 8
fi

