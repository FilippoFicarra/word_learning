#!/bin/bash
# module load eth_proxy gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 cudnn/8.8.1.3 # use this on Euler
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt