#! /bin/bash

for i in homogeneous-PIRING-without-modulation-or-noise.yaml homogeneous-PIRING-with-noise-no-modulation.yaml homogeneous-PIRING-with-noise-and-modulation.yaml homogeneous-PIRING-with-noise-and-modulation-heteroIapp.yaml heterogeneous-PIRING-with-modulation-and-noise.yaml heterogeneous-PIRING-without-modulation-or-noise.yaml
do
    python main.py $i --not-shown
    python graphics.py ${i%%.yaml}.npz --export-video
done
