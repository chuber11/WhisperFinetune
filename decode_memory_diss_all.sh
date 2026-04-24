
#for testset in earnings librispeech_asr.CLEAN librispeech_asr.OTHER yodas
#do
#    for distr in 0 10 100 250 -250 -100 -10 -1
#    do
#        bash decode_memory.sh saves/model_newwords18_2/checkpoint-94000 $distr EN english $testset
#    done
#done

while IFS='|' read -r testset distr
do
    echo Decoding $testset with $distr distractors
    bash decode_memory.sh saves/model_newwords15/checkpoint-184000 $distr EN english $testset
done < <(
    for testset in earnings librispeech_asr.CLEAN librispeech_asr.OTHER yodas
    do
        for distr in 0 10 100 250 -250 -100 -10 -1
        do
            echo "$testset|$distr"
        done | shuf
    done
)

