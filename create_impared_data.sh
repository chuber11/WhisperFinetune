
version=${1:-2}

mkdir -p data_impairedSpeech_new$version
cp /project/data_asr/TEQST-Audio-data/ReadingTexts_DE/german_v2.stm data_impairedSpeech_new$version

cat data_impairedSpeech_new$version/german_v2.stm | grep -v ";;" | awk -v version="$version" '{print $1" data_impairedSpeech_new"version"/"$2".mp3 "$5" "$6}' | grep "text_1-usr0227" > data_impairedSpeech_new$version/impairedSpeech.DE.test.seg.aligned
cat data_impairedSpeech_new$version/german_v2.stm | grep -v ";;" | grep "text_1-usr0227" | cut -d" " -f8- > data_impairedSpeech_new$version/impairedSpeech.DE.test.cased

cat data_impairedSpeech_new$version/german_v2.stm | grep -v ";;" | awk -v version="$version" '{print $1" data_impairedSpeech_new"version"/"$2".mp3 "$5" "$6}' | grep -v "text_1-usr0227" | grep "text_0-usr0227" > data_impairedSpeech_new$version/impairedSpeech.DE.dev.seg.aligned
cat data_impairedSpeech_new$version/german_v2.stm | grep -v ";;" | grep -v "text_1-usr0227" | grep "text_0-usr0227" | cut -d" " -f8- > data_impairedSpeech_new$version/impairedSpeech.DE.dev.cased

cat data_impairedSpeech_new$version/german_v2.stm | grep -v ";;" | awk -v version="$version" '{print $1" data_impairedSpeech_new"version"/"$2".mp3 "$5" "$6}' | grep -v "text_1-usr0227" | grep -v "text_0-usr0227" > data_impairedSpeech_new$version/impairedSpeech.DE.train.seg.aligned
cat data_impairedSpeech_new$version/german_v2.stm | grep -v ";;" | grep -v "text_1-usr0227" | grep -v "text_0-usr0227" | cut -d" " -f8- > data_impairedSpeech_new$version/impairedSpeech.DE.train.cased

cd /project/data_asr/TEQST-Audio-data/ReadingTexts_DE/AudioData

for x in `ls *.wav`
do
    if [ ! -s "$x" ]; then
        continue
    fi
    outfile="/export/data2/chuber/2024/Whisper+Memory/data_impairedSpeech_new$version/${x%.wav}.mp3"
    if [ -e "$outfile" ]; then
        continue
    fi
    echo $x
    ffmpeg -i /project/data_asr/TEQST-Audio-data/ReadingTexts_DE/AudioData/$x -ar 16000 -ac 1 "$outfile"
done

cd -

