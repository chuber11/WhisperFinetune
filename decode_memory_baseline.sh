
model=${1:-whisper-large-v3}
distractors=${2:-0}
language=${3:-EN}
language2=${4:-english}

memory="data_filtered_test/cv_memory.$language.test.words"
memory2="data_filtered_test/cv_memory.$language.test.allwords"

model2=`echo $model | sed "s/\//_/g"`

segfile="data_filtered_test/cv_memory.$language.test.seg.aligned"
hypofile="hypos_memory/$model2.$language.`echo $memory | sed "s/\//_/g"`.$distractors.hyp"

language3=`echo $language | tr '[:upper:]' '[:lower:]'`
SERVER="http://192.168.0.72:5008/asr/infer/$language3,$language3"

rm -f $hypofile
mapfile -t lines < "$segfile"

i=1
for line in "${lines[@]}"; do
    id=`echo $line | cut -d" " -f1`
    mp3=`echo $line | cut -d" " -f2`

    ffmpeg -hide_banner -loglevel error -y -i $mp3 -acodec pcm_s16le -ac 1 -ar 16000 tmp.wav
    wav="tmp.wav"

    prefix="Word which might occur are: `cat $memory | head -n$i | tail -n1 | sed "s/|/, /g" `."

    hypo=`python request_server.py $SERVER $wav "$prefix"`
    echo $id $hypo >> $hypofile

    i=$(($i+1))
done

rm tmp.wav

