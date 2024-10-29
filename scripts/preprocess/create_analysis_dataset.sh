#!/bin/bash
echo "Preprocess analysis datasets"

python3 ./src/preprocess/create_dataset.py config="./configs/preprocess/run_create_analysis_dataset.yaml"