#! /bin/bash

WORKDIR='inf-test/scratch'

[ -d ${WORKDIR} ]||{ mkdir -p ${WORKDIR} || exit 0 ; }

run_and_compute(){
    MODEL=${1}
    TEST=${2}
    RUNNER=${WORKDIR}/$(basename ${MODEL%%.yaml}-${TEST})
    python main.py ${MODEL} ${RUNNER}.npz | tee ${RUNNER}.log && python computeMI.py --feature-array=features --remove-first=100 --shuffle-iterations 1 --output ${MODEL%%.yaml}-${TEST}.json ${RUNNER}.npz | tee -a ${RUNNER}.log 
}

for MODEL in inf-test/*.yaml
do
    pids=()
    for TEST in $(seq 0 4)
    do
       run_and_compute  ${MODEL} ${TEST} & pids+=($!)
    done
    for pid in ${pids[*]}; do wait $pid; done

    pids=()
    for TEST in $(seq 5 9)
    do
       run_and_compute  ${MODEL} ${TEST} & pids+=($!)
    done
    for pid in ${pids[*]}; do wait $pid; done

done

#cd inf-test && python plot-all-runs.py
