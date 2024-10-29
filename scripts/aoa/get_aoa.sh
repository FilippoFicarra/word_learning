#!/bin/bash
echo -e "Extract AoA\n"

echo "Combining dataframes"
python3 ./src/signatures/combine_dataframes.py
python3 ./src/aoa/get_aoa.py config="./configs/aoa/get_aoa.yaml"
python3 ./src/signatures/combine_dataframes.py
