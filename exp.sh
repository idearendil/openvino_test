#!/bin/bash

for var in `seq 1 50`;
do
    echo ${var}
    python CN_test_cpu.py
    python CN_test_openvino.py
done