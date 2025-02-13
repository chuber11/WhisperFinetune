
model=${1:-openai/whisper-large-v2}
beam=${2:-4}
path=`echo $model | sed "s/\//_/g"`
write_at_end="" # "--no_write_at_end"

language=english

python decode_asr.py --segfiles /export/data2/chuber/2024/YTData/test_final.seg.aligned --model_path $model --model_name saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350 --hypo_file hypos/hypo_${path}_beam$beam.numbers_youtube.txt --num_beams $beam --language $language $write_at_end --batch_size 4

#bash decode_numbers_youtube.sh saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1.0003e-6_train_emb0/checkpoint-900
#bash decode_numbers_youtube.sh saves/model_numbers_batchweighting0_fact0_freeze0_real_dev_data0_lr1e-5_train_emb0/checkpoint-350
#python decode_mt.py --segfiles hypos/hypo_openai_whisper-large-v2_beam4.numbers_youtube.txt --model_path saves/model_segmenter1/checkpoint-18000 --hypo_file hypos/hypo_model_segmenter1_checkpoint-18000.numbers_youtube.txt --batch_size 4
#python decode_mt.py --segfiles hypos/hypo_openai_whisper-large-v2_beam4.numbers_youtube.txt --model_path saves/model_segmenter2/checkpoint-480 --hypo_file hypos/hypo_model_segmenter2_checkpoint-18000.numbers_youtube.txt --batch_size 4
#python decode_mt.py --segfiles hypos/hypo_openai_whisper-large-v2_beam4.numbers_youtube.txt --model_path saves/model_segmenter2_moredata/checkpoint-870 --hypo_file hypos/hypo_model_segmenter2_moredata_checkpoint-18000.numbers_youtube.txt --batch_size 4

