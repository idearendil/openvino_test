#!/bin/bash

for var in `seq 1 50`;
do
    echo ${var}
    python SD2_test_gpu.py
    python SD2_test_cpu.py
    python SD2_test_openvino.py
done