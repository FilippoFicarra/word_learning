#!/bin/bash
echo -e "Compute corpus surprisal!\n"

context_type="positive"
model="gpt2"
seed=42
dataset="unified"

while getopts ":t:m:s:d:" opt; do
  case ${opt} in
    t ) context_type=$OPTARG ;;
    m ) model=$OPTARG ;;
    s ) seed=$OPTARG ;;\
    d ) dataset=$OPTARG ;;
    \? ) echo "Usage: $0 -t <context_type> -m <model> -s <seed>"; exit 1 ;;
    : ) echo "Invalid option: $OPTARG requires an argument"; exit 1 ;;
  esac
done
shift $((OPTIND -1))


echo -e "context_type: ${context_type}\nModel: ${model}\nSeed: ${seed}\nDataset: ${dataset}\n"

python3 ./src/signatures/get_corpus_surprisal.py config="./configs/signatures/run_corpus_surprisal_${model}.yaml" seed=${seed} context_type=${context_type} dataset=${dataset}