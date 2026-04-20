# HGT-Model Data Processing Pipeline

## Overview
This repository contains an end-to-end data processing pipeline designed to map raw vehicle trajectories from the pNEUMA dataset to an OpenStreetMap (OSM) network. It processes the raw data through several stages to extract traversal metrics, aggregate traffic states, and ultimately generate rich, ego-centric spatial features suitable for training machine learning models for Connected Autonomous Vehicles (CAVs).

## Pipeline Architecture
The pipeline consists of a series of Python scripts that process the data sequentially:

1. **Network Construction (`build_adjacency.py`)**: Downloads the OSM road network for a specified bounding box, fills missing lane data, and constructs a robust topological adjacency map.
2. **Map Matching & Lane Detection (`map_matching.py`)**: Maps raw GPS trajectories to the OSM network using a Hidden Markov Model (HMM) and dynamically calculates discrete lanes based on vehicle positions.
3. **Network Unification (`unify_networks.py`)**: Consolidates multiple map-matched networks into a single unified dataset, resolving any discrepancies in lane counts.
4. **Network Pruning (`prune_network.py`)**: Cleans the unified network by pruning segments based on specific criteria (e.g., completely un-traversed or disconnected segments).
5. **Traversal Metrics Extraction (`extract_traversal_metrics.py`)**: Computes traversal speeds, stop durations, and relative traffic state metrics to quantify segment congestion.
6. **Traffic State Aggregation (`aggregate_traffic_states.py`)**: Converts discrete vehicle traversals into continuous, time-aggregated traffic states using Time-Exponential Moving Average (T-EMA).
7. **Ego-Centric Feature Extraction (`feature_extractor.py`)**: Projects neighboring background vehicles into the Frenet frame (relative longitudinal/lateral distances) of sampled CAVs and calculates local context features across 6 surrounding zones.

## Running the Pipeline


```bash
# 1. Build Base Topology (example bbox)
python build_adjacency.py --bbox 23.71317 37.97161 23.74515 37.99880 --merge-short

# 2. Map Match Trajectories
python map_matching.py pNEUMA_dataset

# 3. Unify Network Definitions
python unify_networks.py --folder processed_data --filter_files

# 4. Prune Network
python prune_network.py

# 5. Extract Traversal Metrics
python extract_traversal_metrics.py --folder processed_data --update_traversals

# 6. Aggregate Macro Traffic States
python aggregate_traffic_states.py --folder processed_data

# 7. Extract Micro Ego-Centric Features
python feature_extractor.py processed_data
```

## Structure
- `pNEUMA_dataset/`: Directory to place raw CSV trajectory data.
- `processed_data/`: Output directory where processed networks, logs, and extracted metrics are stored.

## Notes
- To test the map matching step on a smaller subset, use the `--test` flag: `python map_matching.py data/pneuma_trajectories.csv --test`
