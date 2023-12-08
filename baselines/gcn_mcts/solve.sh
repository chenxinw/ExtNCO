#!/bin/bash
rm -r ./code/TSP.o
rm -r ./test
make

# rei
#Inst_Num=2
#idx=0
#res="./results/rei/"
#data="./data/rei/"
#Names="TSP-50000.txt"
#City_Nums=50000
#resFile="$res$Names"
#dataFile="$data$Names"
#echo "./test ${idx} ${resFile} ${dataFile} ${City_Nums} ${Inst_Num}"
#./test ${idx} ${resFile} ${dataFile} ${City_Nums} ${Inst_Num}

## tsplib
Inst_Num=1
idx=0
res="./results/tsplib/"
data="./data/tsplib/"
Names=("dsj1000.txt" "pr1002.txt" "u1060.txt" "vm1084.txt" "pcb1173.txt" "d1291.txt" "rl1304.txt" "rl1323.txt" "nrw1379.txt" "u1432.txt" "d1655.txt" "vm1748.txt" "u1817.txt" "rl1889.txt" "d2103.txt" "u2152.txt" "u2319.txt" "pr2392.txt" "pcb3038.txt" "fnl4461.txt" "rl5915.txt" "rl5934.txt" "pla7397.txt")
City_Nums=(1000 1002 1060 1084 1173 1291 1304 1323 1379 1432 1655 1748 1817 1889 2103 2152 2319 2392 3038 4461 5915 5934 7397)
for i in {0..22}
#Names=("rl11849.txt" "usa13509.txt" "brd14051.txt" "d15112.txt" "d18512.txt" "pla33810.txt")
#City_Nums=(11849 13509 14051 15112 18512 33810)
#for i in {0..5}
do
    resFile="$res${Names[i]}"
    dataFile="${data}${Names[i]}"
    echo "./test ${idx} ${resFile} ${dataFile} ${City_Nums[i]} ${Inst_Num}"
    ./test ${idx} ${resFile} ${dataFile} ${City_Nums[i]} ${Inst_Num}
done

## vlsi
#Inst_Num=1
#idx=0
#res="./results/vlsi/"
#data="./data/vlsi/"
##Names=("icx28698.txt" "frh19289.txt" "fnc19402.txt" "xmc10150.txt" "xvb13584.txt" "pjh17845.txt" "xia16928.txt" "icx28698.txt" "fyg28534.txt" "ird29514.txt" "irx28268.txt" "boa28924.txt" "xrh24104.txt" "ics39603.txt" "bby34656.txt" "pba38478.txt" "fry33203.txt" "xib32892.txt" "pbh30440.txt" "fht47608.txt" "fna52057.txt" "bna56769.txt" "dan59296.txt" "lsb22777.txt" "fma21553.txt" "bbz25234.txt" "ido21215.txt" "rbz43748.txt")
##City_Nums=(28698 19289 19402 10150 13584 17845 16928 28698 28534 29514 28268 28924 24104 39603 34656 38478 33203 32892 30440 47608 52057 56769 59296 22777 21553 25234 21215 43748)
##for i in {0..26}
#Names=("fna52057.txt" "bna56769.txt" "dan59296.txt")
#City_Nums=(52057 56769 59296)
#for i in {0..2}
#do
#    resFile="$res${Names[i]}"
#    dataFile="${data}${Names[i]}"
#    echo "./test ${idx} ${resFile} ${dataFile} ${City_Nums[i]} ${Inst_Num}"
#    ./test ${idx} ${resFile} ${dataFile} ${City_Nums[i]} ${Inst_Num}
#done
