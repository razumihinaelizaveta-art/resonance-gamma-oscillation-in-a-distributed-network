#! /bin/bash

run_and_compute(){
    MODEL=${1}
    TEST=${2}
    python main.py ${MODEL} ${MODEL%%.yaml}-${TEST}.npz | tee ${MODEL%%.yaml}-${TEST}.log && python computeMI.py --feature-array=features --remove-first=100 --shuffle-iterations 1 --output ${MODEL%%.yaml}-${TEST}.json ${MODEL%%.yaml}-${TEST}.npz | tee -a ${MODEL%%.yaml}-${TEST}.log 
}

for MODEL in inf-test/*.yaml
do
    pids=()
    for TEST in $(seq 0 9)
    do
       run_and_compute  ${MODEL} ${TEST} & pids+=($!)
    done
    for pid in ${pids[*]}; do wait $pid; done
done

cd inf-test && python plot-all-runs.py
