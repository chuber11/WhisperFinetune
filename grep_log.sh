
key=${2:-ppl_ntp_all}
n=${3:-3}

if [[ $key == *"acc"* ]]; then
    cat $1 | grep eval_loss | tr "," "\n" | grep $key | nl | sort -n -k3,3 | tail -n$n && echo "" && cat $1 | grep eval_loss | tr "," "\n" | grep $key | nl | tail -n1
else
    cat $1 | grep eval_loss | tr "," "\n" | grep $key | nl | sort -r -n -k3,3 | tail -n$n && echo "" && cat $1 | grep eval_loss | tr "," "\n" | grep $key | nl | tail -n1
fi

