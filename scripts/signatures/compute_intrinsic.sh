#!/bin/bash
echo -e "Compute Intrinsic\n"


model="gpt2"
seed=42
dataset="unified"
context_type="positive"

while getopts ":t:m:s:d:" opt; do
  case ${opt} in
    t ) context_type=$OPTARG ;;
    m ) model=$OPTARG ;;
    s ) seed=$OPTARG ;;
    d ) dataset=$OPTARG ;;
    \? ) echo "Usage: $0 -t <context_type> -m <model> -s <seed> -d <dataset>"; exit 1 ;;
    : ) echo "Invalid option: $OPTARG requires an argument"; exit 1 ;;
  esac
done
shift $((OPTIND -1))


echo -e "Model: ${model}\nSeed: ${seed}\nDataset: ${dataset}\nContext type: ${context_type}\n"

python3 ./src/signatures/get_intrinsic.py config="./configs/signatures/run_intrinsic_${model}.yaml" seed=${seed} dataset=${dataset} context_type=${context_type}