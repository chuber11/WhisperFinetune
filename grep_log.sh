
key=${2:-ppl_ntp_all}

cat $1 | grep eval | tr "," "\n" | grep $key | nl | sort -n -k3,3 | tail -n5 && echo "" && cat $1 | grep eval | tr "," "\n" | grep $key | nl | tail -n1

