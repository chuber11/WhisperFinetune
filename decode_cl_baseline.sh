
model=${1:-openai/whisper-large-v2}
beam=${2:-4}
lang=EN
lang2=english

for segfilepath in ../NewsData/segfiles/*.seg.aligned
do
    segfile=${segfilepath#../NewsData/segfiles/}
    hypofile=CL/cl_baseline/${segfile%.seg.aligned}.hyp
    if [ -e "$hypofile" ]; then
        echo "File $hypofile exists."
        continue
    fi
    python decode_asr.py --segfiles $segfilepath --model_path $model --model_name openai/whisper-large-v2 --hypo_file $hypofile --num_beams $beam --language $lang2 --batch_size 8
done

