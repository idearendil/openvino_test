#!/bin/bash

for var in `seq 1 50`;
do
    echo ${var}
    python CN_test_cpu_edge_depth.py
    python CN_test_openvino_edge_depth.py
done