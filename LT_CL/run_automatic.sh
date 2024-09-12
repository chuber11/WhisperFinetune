
#crontab -e
#0 6 * * * bash /export/data2/chuber/2024/Whisper+Memory/LT_CL/run_automatic.sh >> /export/data2/chuber/2024/Whisper+Memory/LT_CL/log.txt 2>&1

source ~/.bashrc

executed=false
for i in {1..72}
do
    usedGPUs=`nvidia-smi | tail -n+39 | head -n-1 | cut -d" " -f5 | sort | uniq | tr "\n" " "`
    echo Used GPUs: $usedGPUs

    for GPU in {0..5}
    do
        if [[ ! "$usedGPUs" == *"$GPU"* ]]
        then
            echo Using GPU: $GPU
            cd /export/data2/chuber/2024/Whisper+Memory/LT_CL
            bash run.sh $GPU alexander.waibel@kit.edu
            #bash run.sh $GPU christian.huber@kit.edu
            executed=true
            break
        fi
    done

    if [ "$executed" = true ]
    then
        break
    fi
    echo No free GPU, waiting...
    sleep 600
done
