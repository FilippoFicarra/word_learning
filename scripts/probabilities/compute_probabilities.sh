#!/bin/bash
echo -e "Comparing surprisal\n"

model_base_path="" # "meta-llama/Meta-Llama-3.1-8B" for llama3
seed=42
model="gpt2"

while getopts ":m:n:s:" opt; do
  case ${opt} in
    m ) model_base_path=$OPTARG ;;
    n) model=$OPTARG ;;
    s ) seed=$OPTARG ;;
    \? ) echo "Usage: $0 -m <model_base_path> -n <model> -s <seed>"; exit 1 ;;
    : ) echo "Invalid option: $OPTARG requires an argument"; exit 1 ;;
    esac
done

if [ -z "$model_base_path" ]; then
  echo "Error: -m <model_base_path> is mandatory"
  echo "Usage: $0 -m <model_base_path> -n <model> -s <seed>"
  exit 1
fi

shift $((OPTIND -1))

echo -e "Model base path: ${model_base_path}\nModel: ${model}\nSeed: ${seed}\n"

python3 ./src/probabilities/get_probabilities.py config="./configs/probabilities/run_probability_${model}.yaml" model_base_path=${model_base_path} seed=${seed}