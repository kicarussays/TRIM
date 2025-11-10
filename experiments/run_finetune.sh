run0(){
python experiments/finetune.py --hospital snuh --rep-type none --ehr   --outcome CVD -d 0 && cd ./
}
run1(){
python experiments/finetune.py --hospital snuh --rep-type none --ehr --signal  --outcome CVD -d 1 && cd ./
}
run2(){
python experiments/finetune.py --hospital snuh --rep-type description --ehr   --outcome CVD -d 2 && cd ./
}
run3(){
python experiments/finetune.py --hospital snuh --rep-type description --ehr --ecg  --outcome CVD -d 3 && cd ./
}
run0 & 
run1 & 
run2 & 
run3