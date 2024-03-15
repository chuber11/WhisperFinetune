
model=${1:-saves/model5/checkpoint-180}
adapter=${2:-None}
beam=${3:-4}
path=`echo $model | sed "s/\//_/g"`

if [ "$adapter" = "None" ]; then
    python decode_asr.py --segfiles data_impairedSpeech/impairedSpeech.DE.test.seg.aligned --model_path $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_beam$beam.txt --num_beams $beam --language german --no_write_at_end --batch_size 8
else
    python decode_asr.py --segfiles data_impairedSpeech/impairedSpeech.DE.test.seg.aligned --model_path openai/whisper-large-v2 --load_adapter_model $model --model_name openai/whisper-large-v2 --hypo_file hypos/hypo_${path}_beam$beam.txt --num_beams $beam --language german --no_write_at_end --batch_size 8
fi

