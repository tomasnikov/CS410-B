#!/usr/bin/env bash

BASE_PATH="datasets/cnn1/"
for b in {10..19}; do
    julia src/cnn.jl "$BASE_PATH$b/" --adversarial --writeCSV
done