#!/bin/sh
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments  10 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments  20 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments  50 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments 100 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments 200 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments 400 --slic_method SLIC  --quiet

python3 model.py --features "avg_color" --dataset stanfordcars --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color centroid" --dataset stanfordcars --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid" --dataset stanfordcars --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid num_pixels" --dataset stanfordcars --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv" --dataset stanfordcars --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv std_deviation_color_hsv" --dataset stanfordcars --graph_type RAG --slic_method SLIC  --quiet