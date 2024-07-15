

for lang in EN DE
do
    echo "---- WER $lang ----"
    echo openai_whisper-large-v2
    bash score_CV.sh openai_whisper-large-v2_cv_filtered_beam4.$lang 2>/dev/null

    echo openai_whisper-large-v2 + gpt4-turbo
    bash score_CV.sh openai_whisper-large-v2_converted_cv_filtered_beam4.$lang 2>/dev/null

    for s in saves/model_segmenter*
    do
        n=`echo $s | cut -c22-`
        echo openai_whisper-large-v2 + textseg$n
        m=`ls -d $s/checkpoint-* | sed "s/\//_/g"`
        bash score_CV.sh ._$m $lang
    done

    for f in `ls scores/* | grep -v cv | grep human | grep -v human_train | sort`
    do
        model_name=`echo ${f:13:-18}`
        echo $model_name
        cat scores/saves_${model_name}_cv_filtered_beam4.$lang.score
    done
done

echo "---- Acc numbers human ----"
echo openai_whisper-large-v2
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_beam4.*.human.txt" test "data_numbers_human/llm_augment_human.*.test.seg.aligned"
echo openai_whisper-large-v2 + gpt4-turbo
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_converted_beam4.*.human.txt" test "data_numbers_human/llm_augment_human.*.test.seg.aligned"
for s in saves/model_segmenter*
do
    n=`echo $s | cut -c22-`
    echo openai_whisper-large-v2 + textseg$n
    m=`echo $s | sed "s/\//_/g"`
    python eval_numbers.py "hypos/hypo_._${m}_checkpoint-*.txt" test "data_numbers_human/llm_augment_human.*.test.seg.aligned"
done

for f in `ls scores/* | grep -v cv | grep human | grep -v human_train | sort`
do
    model_name=`echo ${f:13:-18}`
    echo $model_name
    cat scores/saves_${model_name}_beam4.human.score
done

echo "---- Acc numbers tts ----"
echo openai_whisper-large-v2
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_beam4.*.txt" test
echo openai_whisper-large-v2 + gpt4-turbo
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_converted_beam4.*.txt" test
for s in saves/model_segmenter*
do
    n=`echo $s | cut -c22-`
    echo openai_whisper-large-v2 + textseg$n
    m=`echo $s | sed "s/\//_/g"`
    python eval_numbers.py "hypos/hypo_._${m}_checkpoint-*.txt" test
done

for f in `ls scores/* | grep -v cv | grep human | grep -v human_train | sort`
do
    model_name=`echo ${f:13:-18}`
    echo $model_name
    cat scores/saves_${model_name}_beam4.score
done

echo "---- Acc numbers human train ----"
echo openai_whisper-large-v2
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_beam4.*.human_train.txt" test "data_numbers_human_train/llm_augment_human_train.*.test.seg.aligned"
echo openai_whisper-large-v2 + gpt4-turbo
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_converted_beam4.*.human_train.txt" test "data_numbers_human_train/llm_augment_human_train.*.test.seg.aligned"
for s in saves/model_segmenter*
do
    n=`echo $s | cut -c22-`
    echo openai_whisper-large-v2 + textseg$n
    m=`echo $s | sed "s/\//_/g"`
    python eval_numbers.py "hypos/hypo_._${m}_checkpoint-*.txt" test "data_numbers_human_train/llm_augment_human_train.*.test.seg.aligned"
done

for f in `ls scores/* | grep -v cv | grep human | grep -v human_train | sort`
do
    model_name=`echo ${f:13:-18}`
    file=scores/saves_${model_name}_beam4.human_train.score
    if [ -e $file ]; then
        echo $model_name
        cat scores/saves_${model_name}_beam4.human_train.score
    fi
done

bash score_numbers.sh > /dev/null

for lang in EN DE
do
    echo ---- WER numbers $lang ----

    for s in saves/model_segmenter*
    do
        n=`echo $s | cut -c22-`
        echo openai_whisper-large-v2 + textseg$n
        tail -n4 scores/saves_openai_whisper-large-v2+textseg${n}_beam4.$lang.wer
    done

    for f in `ls scores/*.$lang.wer`
    do
        model_name=`echo ${f:13:-13}`
        lang=`echo $f | rev | cut -d"." -f2 | rev`
        echo $model_name
        file=scores/saves_${model_name}_beam4.$lang.wer
        tail -n4 $file
    done
done

#python scores_to_table.py

