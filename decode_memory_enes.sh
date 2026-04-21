
model=${1:-enes3/checkpoint-52000}
testset=${2:-talcs}
lang=${3:-ZH}
lang2=${4:-chinese}
words=${5:-oracle}

# exit if data_test_enes/enes_${testset}_memory.${lang}.words exists
if [ -f "data_test_enes/enes_${testset}_memory.${lang}.test.words" ]; then
    echo "data_test_enes/enes_${testset}_memory.${lang}.test.words exists, exiting"
    exit 1
fi

mv data_test_enes/enes_${testset}_memory.${lang}.test.words_$words data_test_enes/enes_${testset}_memory.${lang}.test.words

model2=`echo $model | sed 's#/#_#g'`

python decode_asr.py --segfiles data_test_enes/enes_${testset}_memory.$lang.test.seg.aligned --model_path saves/model_newwords18_2_$model --model_name saves/model_newwords18_2_$model --use_memory --memory_file data_test_enes/enes_${testset}_memory.$lang.test.allwords --language $lang2 --hypo_file hypos_memory_enes/${model2}_${testset}_${lang}_${words}.hyp --no_write_at_end --memory_num_distractors 0 --batch_size 1 --force_exact_memory 0

mv data_test_enes/enes_${testset}_memory.${lang}.test.words data_test_enes/enes_${testset}_memory.${lang}.test.words_$words

