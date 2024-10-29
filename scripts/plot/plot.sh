#!/bin/bash
mode="summary"

while getopts ":m:" opt; do
  case ${opt} in
    m ) mode=$OPTARG ;;
    \? ) echo "Usage: $0 [-m <mode>]"; exit 1 ;;
    : ) echo "Invalid option: $OPTARG requires an argument"; exit 1 ;;
  esac
done
shift $((OPTIND -1))

echo "Combining dataframes"
python3 ./src/signatures/combine_dataframes.py
echo "Plotting the results\n"
python3 ./src/plotter/plot.py config="configs/plotter/plot.yaml" --m ${mode}