
cat data/german_v2.stm | grep -v ";;" | awk '{print $1" data/"$2".mp3 "$5" "$6}' | grep "text_1-usr0227" > data/impairedSpeech.DE.test.seg.aligned
cat data/german_v2.stm | grep -v ";;" | grep "text_1-usr0227" | cut -d" " -f8- > data/impairedSpeech.DE.test.cased

cat data/german_v2.stm | grep -v ";;" | awk '{print $1" data/"$2".mp3 "$5" "$6}' | grep -v "text_1-usr0227" | grep "text_0-usr0227" > data/impairedSpeech.DE.dev.seg.aligned
cat data/german_v2.stm | grep -v ";;" | grep -v "text_1-usr0227" | grep "text_0-usr0227" | cut -d" " -f8- > data/impairedSpeech.DE.dev.cased

cat data/german_v2.stm | grep -v ";;" | awk '{print $1" data/"$2".mp3 "$5" "$6}' | grep -v "text_1-usr0227" | grep -v "text_0-usr0227" > data/impairedSpeech.DE.train.seg.aligned
cat data/german_v2.stm | grep -v ";;" | grep -v "text_1-usr0227" | grep -v "text_0-usr0227" | cut -d" " -f8- > data/impairedSpeech.DE.train.cased
