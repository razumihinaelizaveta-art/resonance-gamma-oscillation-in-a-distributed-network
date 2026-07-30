#! /bin/bash

for MODEL in inf-test/*.yaml
do
    python main.py ${MODEL} ${MODEL%%.yaml}.npz | tee ${MODEL%%.yaml}.log
done

