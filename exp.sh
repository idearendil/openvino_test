#!/bin/bash

for var in `seq 1 50`;
do
    echo ${var}
    python CN_test_openvino.py
    python CN_test_openvino_edge.py
    python CN_test_openvino_depth.py
done