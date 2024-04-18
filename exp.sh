#!/bin/bash

for var in `seq 1 50`;
do
    echo ${var}
    python YOLOv8_cpu.py
    python YOLOv8_openvino.py
    python YOLOv8_openvino_int8.py
done