
for model in enes3/checkpoint-52000 enes3_2/checkpoint-150000 enes3_3/checkpoint-134000 enes3_4/checkpoint-107000 enes3_5/checkpoint-127000
do
    for testset in "talcs ZH chinese"
    do
        bash decode_memory_enes.sh $model $testset oracle
    done
done

