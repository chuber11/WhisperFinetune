
model=tiny
path="saves/model_confidence_tiny/checkpoint-75000/full"
python decode_asr.py --segfiles confidence/output_combined/dev.txt --model_name openai/whisper-$model --model_path $path --eval_confidence --hypo_file hypos_confidence/dev_$model.txt --no_write_at_end --batch_size 32

model=small
path="saves/model_confidence_small/checkpoint-69000/full"
python decode_asr.py --segfiles confidence/output_combined/dev.txt --model_name openai/whisper-$model --model_path $path --eval_confidence --hypo_file hypos_confidence/dev_$model.txt --no_write_at_end --batch_size 32

model=medium
path="saves/model_confidence_medium/checkpoint-?/full"
#python decode_asr.py --segfiles confidence/output_combined/dev.txt --model_name openai/whisper-$model --model_path $path --eval_confidence --hypo_file hypos_confidence/dev_$model.txt --no_write_at_end --batch_size 32

model=large-v2
path="saves/model_confidence_large-v2-cont/checkpoint-8000/full"
python decode_asr.py --segfiles confidence/output_combined/dev.txt --model_name openai/whisper-$model --model_path $path --eval_confidence --hypo_file hypos_confidence/dev_$model.txt --no_write_at_end --batch_size 32

