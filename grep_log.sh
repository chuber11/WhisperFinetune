
key=${2:-ppl_ntp_all}
n=${3:-3}
key2=${4:-eval_loss}

if [[ $key == *"acc"* ]]; then
    cat $1 | grep $key2 | grep -v "metric_for_best_model" | tr "," "\n" | grep $key | nl | sort -n -k3,3 | tail -n$n && echo "" && cat $1 | grep $key2 | grep -v "metric_for_best_model" | tr "," "\n" | grep $key | nl | tail -n1
else
    cat $1 | grep $key2 | tr "," "\n" | grep $key | grep -v "metric_for_best_model" | nl | sort -r -n -k3,3 | tail -n$n && echo "" && cat $1 | grep $key2 | grep -v "metric_for_best_model" | tr "," "\n" | grep $key | nl | tail -n1
fi

