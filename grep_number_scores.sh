

for lang in EN DE
do
    echo "---- WER $lang ----"
    echo hypos/hypo_openai_whisper-large-v2
    bash score_CV.sh openai_whisper-large-v2_cv_filtered_beam4.$lang 2>/dev/null

    for f in `ls scores/* | grep -v cv | grep human | sort`
    do
        model_name=`echo ${f:13:-18}`
        echo $model_name
        cat scores/saves_${model_name}_cv_filtered_beam4.$lang.score
    done
done

echo "---- Acc numbers human ----"
echo hypos/hypo_openai_whisper-large-v2
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_beam4.*.human.txt" test "data_numbers_human/llm_augment_human.*.test.seg.aligned"
echo hypos/hypo_openai_whisper-large-v2_converted
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_converted_beam4.*.human.txt" test "data_numbers_human/llm_augment_human.*.test.seg.aligned"

for f in `ls scores/* | grep -v cv | grep human | sort`
do
    model_name=`echo ${f:13:-18}`
    echo $model_name
    cat scores/saves_${model_name}_beam4.human.score
done

echo "---- Acc numbers tts ----"
echo hypos/hypo_openai_whisper-large-v2
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_beam4.*.txt" test
echo hypos/hypo_openai_whisper-large-v2_converted
python eval_numbers.py "hypos/hypo_openai_whisper-large-v2_converted_beam4.*.txt" test
for f in `ls scores/* | grep -v cv | grep human | sort`
do
    model_name=`echo ${f:13:-18}`
    echo $model_name
    cat scores/saves_${model_name}_beam4.score
done

python scores_to_table.py

