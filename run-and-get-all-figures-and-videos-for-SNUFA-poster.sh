#! /bin/bash

python main.py homogeneous-PIRING-without-modulation-or-noise.yaml --not-shown
python graphics.py homogeneous-PIRING-without-modulation-or-noise.npz --export-video

python main.py homogeneous-PIRING-with-noise-no-modulation.yaml --not-shown
python graphics.py homogeneous-PIRING-with-noise-no-modulation.npz --export-video

python main.py homogeneous-PIRING-with-noise-and-modulation.yaml --not-shown
python graphics.py homogeneous-PIRING-with-noise-and-modulation.npz --export-video

python main.py homogeneous-PIRING-with-noise-and-modulation-heteroIapp.yaml --not-shown
python graphics.py homogeneous-PIRING-with-noise-and-modulation-heteroIapp.npz --export-video

