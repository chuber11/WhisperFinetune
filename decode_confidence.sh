
python decode_asr.py --segfiles confidence/output_combined/dev.txt --model_name openai/whisper-tiny --model_path saves/model_confidence_tiny/checkpoint-75000/full --eval_confidence --hypo_file hypos_confidence/dev_tiny.txt --no_write_at_end --batch_size 64

#python decode_asr.py --segfiles confidence/output_combined/dev.txt --model_name openai/whisper-tiny --model_path saves/model_confidence_small/checkpoint-14000/full --eval_confidence --hypo_file hypos_confidence/dev_small.txt --no_write_at_end --batch_size 64

