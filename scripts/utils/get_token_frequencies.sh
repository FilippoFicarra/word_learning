#!/bin/bash
echo -e "Get token frequencies\n"

python3 ./src/utils/get_token_frequencies.py config="./configs/utils/get_token_frequencies.yaml"