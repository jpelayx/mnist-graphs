#!/bin/sh

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments  10 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments  20 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments  50 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments 100 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments 200 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments 400 --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments  10 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments  20 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments  50 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments 100 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments 200 --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --n_segments 400 --slic_method SLIC  --quiet

python3 model.py --features "avg_color" --dataset geo_ds --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color centroid" --dataset geo_ds --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid" --dataset geo_ds --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid num_pixels" --dataset geo_ds --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv" --dataset geo_ds --graph_type RAG --slic_method SLIC  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid avg_color_hsv std_deviation_color_hsv" --dataset geo_ds --graph_type RAG --slic_method SLIC  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 1NNSpatial --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 2NNSpatial --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 4NNSpatial --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 8NNSpatial --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 16NNSpatial --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 1NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 2NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 4NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 8NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type 16NNFeature --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset geo_ds --graph_type RAG --slic_method SLIC0  --quiet



python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments  10 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments  20 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments  50 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments 100 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments 200 --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --n_segments 400 --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 1NNSpatial --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 2NNSpatial --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 4NNSpatial --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 8NNSpatial --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 16NNSpatial --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 1NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 2NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 4NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 8NNFeature --slic_method SLIC0  --quiet
python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type 16NNFeature --slic_method SLIC0  --quiet

python3 model.py --features "avg_color std_deviation_color centroid std_deviation_centroid" --dataset stanfordcars --graph_type RAG --slic_method SLIC0  --quiet
