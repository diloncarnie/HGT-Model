Act as a Senior Machine Learning Engineer and Spatial Data Scientist. We are building a highly optimized, preprocessing pipeline for the pNEUMA trajectory dataset to prepare it for a temporal Graph Neural Network (GNN). 

Your job is to first create a plan to modify and expand the existing scripts to achieve the tasks below. When you have come up with a robust plan, you will explain it step by step with no gaps. Then you can proceed to create/modify the scripts in the current directory so that it can interact with the dataset stored in the pNEUMA_dataset folder. **Only proceed to execute the plan after I have accepted it.**

You must write/modify two separate Python scripts: extract_map_and_speeds.py (Initialization) and preprocess_pneuma.py (Main Pipeline). Use standard libraries (pandas, numpy, scipy.spatial, scipy.interpolate, jenkspy, leuvenmapmatching, osmnx, shapely, geopandas, matplotlib.pyplot, argparse, multiprocessing, time).

**CRITICAL ARCHITECTURE & SETUP RULES:**
- **Dataset Context:** The dataset is stored in pNEUMA_dataset in the root of the current directory. There are 20 folders covering 4 days split into 5 different time intervals.
- **CLI & I/O:** Use argparse to accept either a directory of CSV files or a specific CSV file.
- **Independence:** **Do NOT mutate track_ids.** Treat every file as a separate dataset.
- **Multiprocessing:** Heavily utilize multiprocessing.Pool for parsing wide-to-long CSV chunks and for processing independent timestamp groupings to manage memory.
- **Deferred Computation:** Do not calculate Ego-only kinematic features (relative_kinematic_ratio, change_in_euclidean_distance, etc.) on the entire dataset. Calculate these ONLY at the very end of the pipeline, after the non-CAV "Ghost Nodes" have been purged.
- **Performance Profiling:** Add explicit time debugging outputs (e.g., using the `time` module) at every main section of processing in BOTH scripts to print how much time each section took (e.g., "Initialization took X seconds", "Map-matching took Y seconds", etc.).

**RAW DATASET STRUCTURE & PARSING RULES:**
- Each Row is a unique vehicle with Cols 1-10: Headers.
- Cols 1-4: Static info (track_id, type, traveled_distance in m, avg_speed in km/h).
- Cols 5+: Repeating 6-col chunks (lat, lon, speed in km/h, lon_acc in m/s², lat_acc in m/s², time in s).
- **CRITICAL:** Convert all speeds (avg_speed and repeating speed) from km/h to m/s immediately during the parsing phase.
- The existing vehicle types and sizes are as follows:
    - Car and Taxi:  5 x 2
    - Medium Vehicle: 5.83 x 2.67
    - Heavy Vehicle: 12.5 x 3.33
    - Bus: 12.5 x 4
    - Motorcycle: 2.5 x 1
- In each script, add a --test flag functionality that only selects the first 15 vehicles (rows) from each provided file for testing.

**EXTERNAL SCRIPT:**

The following  python script exists in the root directory and is used to download the Athens road network and output a topological_adjacency.json which maps road segments to their sucessors and predecessors. It will be called at the start and the .gpkg and .json files will be stored in the root directory to be accessed by Script 1 and 2:

```python
import geopandas as gpd
import pandas as pd
import osmnx as ox
import math
import json
import argparse
import os
import numpy as np

def get_osm_network():
    print("Downloading OSM network for Athens...")
    try:
        graph = ox.graph_from_place('Athens, Greece', network_type='drive')
    except Exception as e:
        # Bounding box covering Athens center
        graph = ox.graph_from_bbox(bbox=(23.70, 37.95, 23.76, 38.00), network_type='drive')

    valid_highway_types = ['primary', 'secondary', 'tertiary', 'trunk']
    edges_to_keep = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        hw = data.get('highway')
        if isinstance(hw, list):
            hw = hw[0]
        if hw in valid_highway_types:
            edges_to_keep.append((u, v, k))
            
    filtered_graph = graph.edge_subgraph(edges_to_keep).copy()
    graph_utm = ox.project_graph(filtered_graph)
    
    nodes, edges = ox.graph_to_gdfs(graph_utm)
    edges = edges.reset_index()
    edges['segment_id'] = edges.index.astype(str)
    
    if 'length' not in edges.columns:
        edges['length'] = edges.geometry.length
    if 'lanes' not in edges.columns:
        edges['lanes'] = 2
    else:
        edges['lanes'] = pd.to_numeric(edges['lanes'], errors='coerce').fillna(2)
        
    def get_first_hw(x):
        return x[0] if isinstance(x, list) else x
    edges['highway'] = edges['highway'].apply(get_first_hw)
    
    return edges

def get_smoothed_heading(geom, reverse=False, look_dist=10.0):
    """
    Calculates a stable heading by looking back/forward 'look_dist' meters 
    along the vertices from the junction node.
    """
    coords = list(geom.coords)
    if len(coords) < 2:
        return 0.0
    
    if not reverse:
        # EXIT HEADING: Look back from the end point (coords[-1])
        ref_pt = coords[-1]
        target_pt = coords[-2]
        accum_dist = 0.0
        for i in range(len(coords) - 1, 0, -1):
            p_end = coords[i]
            p_start = coords[i-1]
            d = math.sqrt((p_end[0]-p_start[0])**2 + (p_end[1]-p_start[1])**2)
            if accum_dist + d >= look_dist:
                target_pt = p_start
                break
            accum_dist += d
            target_pt = p_start
        p1, p2 = target_pt, ref_pt
    else:
        # ENTRY HEADING: Look ahead from the start point (coords[0])
        ref_pt = coords[0]
        target_pt = coords[1]
        accum_dist = 0.0
        for i in range(0, len(coords) - 1):
            p_start = coords[i]
            p_end = coords[i+1]
            d = math.sqrt((p_end[0]-p_start[0])**2 + (p_end[1]-p_start[1])**2)
            if accum_dist + d >= look_dist:
                target_pt = p_end
                break
            accum_dist += d
            target_pt = p_end
        p1, p2 = ref_pt, target_pt
        
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def build_topological_adjacency(edges_gdf):
    print("Building topological lookups...")
    # Grouping by node IDs
    edges_by_u = edges_gdf.groupby('u')['segment_id'].apply(list).to_dict()
    edges_by_v = edges_gdf.groupby('v')['segment_id'].apply(list).to_dict()
    
    adjacency = {}
    
    print("Pre-calculating smoothed headings (10m window)...")
    segment_data = {}
    for _, row in edges_gdf.iterrows():
        sid = str(row['segment_id'])
        geom = row['geometry']
        segment_data[sid] = {
            'u': row['u'],
            'v': row['v'],
            'osmid': str(row['osmid']),
            'exit_heading': get_smoothed_heading(geom, reverse=False),
            'entry_heading': get_smoothed_heading(geom, reverse=True)
        }

    print("Analyzing topological connections...")
    for ego_id, data in segment_data.items():
        u_node, v_node = data['u'], data['v']
        ego_osmid = data['osmid']
        
        valid_successors = []
        valid_predecessors = []
        
        # 1. SUCCESSORS: segments starting where we end (v_node)
        potential_successors = edges_by_u.get(v_node, [])
        for succ_id in potential_successors:
            succ_id_str = str(succ_id)
            if succ_id_str == ego_id: continue
            
            # Heading Check
            succ_heading = segment_data[succ_id_str]['entry_heading']
            delta_theta = (succ_heading - data['exit_heading'] + 180) % 360 - 180
            
            # Same OSM Way Check (Priority)
            is_same_way = segment_data[succ_id_str]['osmid'] == ego_osmid
            threshold = 60.0 if is_same_way else 45.0
            
            if abs(delta_theta) <= threshold:
                valid_successors.append(succ_id_str)
                
        # 2. PREDECESSORS: segments ending where we start (u_node)
        potential_predecessors = edges_by_v.get(u_node, [])
        for pred_id in potential_predecessors:
            pred_id_str = str(pred_id)
            if pred_id_str == ego_id: continue
            
            pred_heading = segment_data[pred_id_str]['exit_heading']
            delta_theta = (data['entry_heading'] - pred_heading + 180) % 360 - 180
            
            is_same_way = segment_data[pred_id_str]['osmid'] == ego_osmid
            threshold = 60.0 if is_same_way else 45.0
            
            if abs(delta_theta) <= threshold:
                valid_predecessors.append(pred_id_str)
                
        adjacency[ego_id] = {
            'successors': valid_successors,
            'predecessors': valid_predecessors
        }
        
    return adjacency

def main():
    parser = argparse.ArgumentParser(description="Build robust topological adjacency. Always downloads the latest network.")
    parser.add_argument('--output', default='topological_adjacency.json', help="Output JSON filename.")
    args = parser.parse_args()
    
    # Always download the network as requested
    edges = get_osm_network()
    network_path = "osm_network.gpkg"
    print(f"Saving frozen OSM network to {network_path}...")
    
    # GPKG cannot handle lists well. Convert any list-like osmid or highway tags to simple strings for saving.
    edges_to_save = edges.copy()
    for col in edges_to_save.columns:
        if edges_to_save[col].apply(lambda x: isinstance(x, list)).any():
            edges_to_save[col] = edges_to_save[col].astype(str)
            
    edges_to_save.to_file(network_path, driver="GPKG")
    
    adjacency = build_topological_adjacency(edges)
    
    print(f"Exporting adjacency to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(adjacency, f, indent=4)
    print("Done.")

if __name__ == '__main__':
    main()
```

### Script 1: extract_map_and_speeds.py (Initialization)
**Objective:** Process a single sample pNEUMA CSV to generate the static foundational map, empirical speeds, and global lane boundaries. (Note: Topological adjacency is handled externally).

**Execution Steps:**
1. **Global Map-Matching:** Parse a single CSV provided to long-format. Drop "Motorcycles". After loading the GPKG network Map-match all remaining vehicles to the UTM network using leuvenmapmatching (Iterative search: start at max_dist=5m, increment by 5m up to 50m on NoMatchError).
3. **Global Lane Boundaries (Jenks):** Calculate the signed orthogonal distance of every vehicle to the matched OSM centerline using the 2D cross-product of the segment vector and point vector. Assign negative values to vehicles on the left, and positive to the right.
   - **Anchor to Right Edge:** Because OSM centerlines are often physically offset, do not trust the centerline as the true center. Find the 98th percentile of the signed distances on that segment to find the "Statistical Maximum" (the true right-most vehicle to compensate for outliers). Subtract every vehicle's signed distance from that maximum (D = Max_Distance - Vehicle_Distance) to yield a clean distance_from_right_edge. Finally, clip any outliers that are further right of the anchor to 0. This completely cancels out any OSM centerline offsets and yields a clean distance_from_right_edge where 0.0 is the true right-most boundary.
   - Group the data by segment_id.
   - **Data-Driven (Jenks):** If a segment has > 50 trajectory points, use jenkspy on the "Car/Taxi" subset (max sample 5,000) to find the lane boundaries. Enforce lane widths between 2.5m and 4.0m.
   - **Static Fallback:** If a segment has < 50 points, generate static geometric boundaries based on the OSM num_lanes (e.g., [0.0, 3.2, 6.4...]).
   - Save the boundary arrays for every segment to lane_boundaries.json.
4. **Debug Histograms:** Randomly select 5 segment_ids that successfully used the Jenks method. Use matplotlib to plot a histogram of their D coordinates, overlaying vertical lines for the calculated Jenks boundaries. Save these 5 plots as PNGs in the output directory.
5. **Empirical Speeds:** Filter the matched trajectories for "Car" and "Medium" Vehicles only. Group by segment_id and track. Calculate traversal speeds. Find the 95th percentile speed per segment. Impute missing segments using default highway speeds (e.g., primary=14m/s, secondary=11m/s, tertiary=9m/s). Export to empirical_free_flow_speeds.json.
6. **Exports:** Ensure that both JSONs (empirical_free_flow_speeds.json, lane_boundaries.json) are saved in the processed_data directory. 

---

### Script 2: preprocess_pneuma.py (Main Pipeline)

I will provide you with a base Python template preprocess_pneuma.py that currently handles basic wide-to-long flattening, temporal downsampling, and multiprocessing for the pNEUMA dataset. 

```python
import multiprocessing
import sys
import time
import os
import csv

chunk_size = 10  # Adjust this based on your system's memory capacity
sampling_interval = 500  # Change this value for different sampling, it's in milliseconds

def process_chunk(chunk_data, chunk_index, output_dir):
    processed_data = []
    start_time = time.time()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    temp_output_file = os.path.join(output_dir, f'temp_chunk_{chunk_index}.csv')
    
    with open(temp_output_file, 'w', newline='') as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for line in chunk_data:
            data = line.strip().split(';')
            track_id = data[0].strip()
            vehicle_type = data[1].strip()
            traveled_d = data[2].strip()
            avg_speed = data[3].strip()
            data_points = data[4:]  # Assuming the first 4 columns are id, type, traveled_d, avg_speed
            
            last_included_timestamp = None

            for i in range(0, len(data_points)-1, 6):
                try:
                    lat, lon, speed, lon_acc, lat_acc, timestamp = data_points[i:i+6]
                    timestamp_ms = int(float(timestamp) * 1000) # Convert to milliseconds for sampling logic
                    
                    if last_included_timestamp is None or timestamp_ms - last_included_timestamp >= sampling_interval:
                        processed_data.append([
                            track_id, vehicle_type, traveled_d, avg_speed, 
                            lat.strip(), lon.strip(), speed.strip(), 
                            lon_acc.strip(), lat_acc.strip(), timestamp.strip()
                        ])
                        last_included_timestamp = timestamp_ms
                except ValueError as e:
                    print(f"ValueError occurred: {e}")
                    print(f"Problematic data: {data_points[i:i+6]}")
                    print(f"In line: {track_id} index {i}")
                    continue

        for row in processed_data:
            csv_writer.writerow(row)

    end_time = time.time()
    if chunk_index % 10 == 0: 
        print(f"Chunk {chunk_index} processed in {end_time - start_time} seconds")

def concatenate_files(output_file, output_dir):
    def sort_key(filename):
        parts = filename.split('_')
        if parts[0] == 'temp' and parts[1] == 'chunk' and parts[2].split('.')[0].isdigit():
            return int(parts[2].split('.')[0])
        return float('inf')  # Put non-matching files at the end

    with open(output_file, 'w', newline='') as f_out:
        # Write headers for the flattened structure
        f_out.write('track_id,type,traveled_d,avg_speed,lat,lon,speed,lon_acc,lat_acc,time\n') 
        
        for filename in sorted(os.listdir(output_dir), key=sort_key):
            if filename.startswith('temp_chunk_'):
                with open(os.path.join(output_dir, filename), 'r') as f_in:
                    lines = f_in.readlines()
                    f_out.writelines(lines) 
                os.remove(os.path.join(output_dir, filename))

def process(input_file: str, output_dir: str) -> None:
    output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.csv', '_processed.csv'))

    with open(input_file, 'r') as file:
        lines = file.readlines()
        # Skip the header line (index 0) and split into chunks
        chunks = [lines[i:i + chunk_size] for i in range(1, len(lines), chunk_size)]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for chunk_index, chunk_data in enumerate(chunks):
        pool.apply_async(process_chunk, args=(chunk_data, chunk_index, output_dir))

    pool.close()
    pool.join()

    concatenate_files(output_file, output_dir)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <input_file> [output_directory]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else os.path.dirname(input_file)

    start_time = time.time()
    process(input_file, output_dir)
    end_time = time.time()
    print(f"File processed in {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
```

You must use this script as your starting template. Modify and expand it according to the instructions below to process raw pNEUMA trajectory CSVs into fully enriched, long-format CSVs. 

**Execution Steps per File:**
1. **Initialization:** Load the GPKG network, speeds JSON, lane boundaries JSON, and the topological_adjacency.json into memory.
2. **Standardization & Ghost Node Prep:** Parse wide-to-long using multiprocessing. Convert speeds from km/h to m/s immediately. 
   - **Class Filter:** Drop "Motorcycles" ONLY. Retain Cars, Taxis, Medium/Heavy Vehicles, and Buses (Ghost Nodes for spatial occupancy).
   - **Duration Filter:** Drop trajectories < 5.0s duration. 
    - **Temporal Prep (Empirical Downsampling):** Utilize the template's sampling logic to retain empirical trajectory points. Set `sampling_interval = 1000` (1.0s). Retain the empirical points where the time gap from the previously retained point is $\ge 1.0s$. This preserves the natural recording variance of the dataset while achieving an average 1.0Hz frequency. Calculate the exact `relative_time_gap` between these newly retained points.
3. **Unified Map-Matching:** Map-match ALL remaining vehicles using the iterative leuvenmapmatching logic. 
    - *Post-Match Filters:* Drop trajectories if the angular difference between the vehicle's vector and the road's vector is > 90 degrees, or if the max orthogonal distance to the edge is > 10m.
    - Extract `segment_id`, `segment_length`, `segment_type`, `num_lanes`.
    - Calculate `raw_offset`. Calculate `proportionate_distance_travelled = raw_offset / segment_length`.
4. **O(1) Lane Assignment:**
   - **Data-Driven Offset Invariance:** Calculate the signed orthogonal distance of every vehicle to the matched OSM centerline using the 2D vector cross-product. Assign negative to the left, positive to the right.
   - **Anchor to Right Edge:** Find the 98th percentile of the signed distances on that segment to find the "Statistical Maximum" (the true right-most vehicle to compensate for outliers). Subtract every vehicle's signed distance from that maximum (D = Max_Distance - Vehicle_Distance) to yield a clean distance_from_right_edge. Finally, clip any outliers that are further right of the anchor to 0.
   - *Optimization:* Instantly assign the discrete lane_index to ALL vehicles by using np.digitize(D, bins=loaded_boundaries[segment_id]).
5. **Asymmetric Ego-Centric Feature Extraction (Frenet Bounding Boxes):**
   - **Stratified Temporal CAV Sampling:** Bin entry times into 5-minute chunks, sample exactly 15% of "Car" and "Taxi" tracks per bin, label as is_CAV=True.
   - At every timestamp (using multiprocessing), build a scipy.spatial.KDTree of ALL vehicles. Query for CAVs.
   - **Frenet Filtering:** Filter neighbors using the topological_adjacency dictionary (must be on the same, successor, or predecessor segment). Enforce same-direction travel (dot product > 0).
   - **Curvilinear Math:** Calculate ΔD (D_neighbor - D_ego). Calculate ΔS adjusting for segment length transitions across downstream/upstream boundaries.
   - **Zone Sorting (CRITICAL ROUTING MATH):** Use the calculated ΔS and ΔD to project filtered neighbors into 4 distinct Ego-Centric zones. Apply these exact conditional bounds (assuming a lane width threshold of ±1.6m from the ego vehicle's center):
     - **Proceeding (Same Lane, Ahead):** 0 < ΔS <= 50 AND -1.6 <= ΔD <= 1.6
     - **Following (Same Lane, Behind):** -50 <= ΔS < 0 AND -1.6 <= ΔD <= 1.6
     - **Leftwards (Adjacent Lanes Left):** -50 <= ΔS <= 50 AND ΔD > 1.6
     - **Rightwards (Adjacent Lanes Right):** -50 <= ΔS <= 50 AND ΔD < -1.6
   - **Zone Calculations:** Calculate raw density and average absolute speed (m/s) per zone. Calculate **Relative Occupancy**: 
     - Proceeding/Following: min(1.0, sum(neighbor_length + 2m gap) / 50m).
     - Left/Right: min(1.0, sum(neighbor_length + 2m gap) / (100m * remaining_lane_count)).
     - *Imputation:* Impute empty zones to a relative speed of 1.0 and occupancy of 0.0.
6. **The Purge & Deferred Kinematics:**
   - Drop all rows where is_CAV == False to purge Ghost Nodes.
   - On this heavily reduced dataframe, calculate change_in_euclidean_distance.
   - Normalize against global speeds: relative_kinematic_ratio = min(1.0, change_in_euclidean_distance / (segment_free_flow_speed * relative_time_gap)).
   - Calculate relative_ego_speed and the 4 relative_speed_[zone] features.
   - Export to a final, long-format CSV.
   - Clean up intermediate files that may have been used for processing.

**Required Final Output Columns:**
track_id, type, traveled_distance, avg_speed, lat, lon, speed, long_acc, lat_acc, time, segment_id, segment_length, segment_type, num_lanes, lane_index, proportionate_distance_travelled, change_in_euclidean_distance, relative_time_gap, relative_kinematic_ratio, relative_occupancy_proceeding, relative_occupancy_following, relative_occupancy_leftwards, relative_occupancy_rightwards, raw_density_proceeding, raw_density_following, raw_density_leftwards, raw_density_rightwards, segment_free_flow_speed, relative_ego_speed, relative_speed_proceeding, relative_speed_following, relative_speed_leftwards, relative_speed_rightwards, raw_speed_proceeding, raw_speed_following, raw_speed_leftwards, raw_speed_rightwards.