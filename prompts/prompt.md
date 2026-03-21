Act as a Senior Machine Learning Engineer and Spatial Data Scientist. I will provide you with a base Python template `preprocess_pneuma.py` that currently handles basic wide-to-long flattening, temporal downsampling, and multiprocessing for the pNEUMA dataset. 

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

You must use this script as your starting template. Modify and expand it according to the instructions below to process raw pNEUMA trajectory CSVs into fully enriched, long-format CSVs optimized for a temporal, heterogeneous Graph Neural Network (GNN). Use standard libraries (`pandas`, `numpy`, `scipy.spatial`, `scipy.interpolate`, `osmnx`, `shapely`, `geopandas`, `argparse`). 

**Template Context & Execution Architecture (CRITICAL):**
- Keep the `multiprocessing` chunk-based approach for the initial wide-to-long parsing to manage memory.
- Use `argparse` to accept either a directory of CSV files or a specific list of CSV files.
- **Do NOT mutate `track_id`s.** Treat every file as a separate dataset.
- Implement a **Two-Pass Architecture**. *Note: Because the KD-Tree (Pass 2) requires all vehicles to be present at a specific timestamp, you must adapt the template to aggregate the parsed chunks before performing spatial grouping.*
  - **Pass 1 (Global Speed Calculation):** Iterate through all specified input files to calculate a single, global empirical free-flow speed for every OSM segment.
  - **Pass 2 (Feature Engineering):** Iterate through the files again to calculate spatial features, normalize using the global speeds, and export each file individually.

**Raw Dataset Structure & Parsing Rules (Already partially in template):**
- Row 1, Cols 1-10: Headers.
- Cols 1-4: Static info (`track_id`, `type`, `traveled_distance` in m, `avg_speed` in km/h).
- Cols 5+: Repeating 6-col chunks (`lat`, `lon`, `speed` in km/h, `lon_acc` in m/s², `lat_acc` in m/s², `time` in s).
- Dataset is stored in pNEUMA_dataset in the root of the current directory. There are 20 folders covering 4 days split into 5 different time intervals.
- **CRITICAL:** Convert all speeds (`avg_speed` and repeating `speed`) from km/h to m/s immediately during the parsing phase.

Execute the pipeline using these exact logical steps:

**PASS 1: Global Free-Flow Speed Derivation**
- Download the drivable OSM network for the Athens bounding box. Filter to retain ONLY `primary`, `secondary`, `tertiary`, and `trunk` (drop motorways, residential, `*_link`). Project to UTM Cartesian (e.g., EPSG:32634).
- For every specified input file:
  - Parse the wide-format CSV into long-format (utilize the template's multiprocessing here). 
  - Filter for ONLY the "Car" vehicle type.
  - Map-match coordinates to the UTM OSM network to find `segment_id` and `raw_offset` (longitudinal distance).
  - Group by `segment_id` and `track_id`. Fit a spline mapping `raw_offset` against `time` for full segment traversals.
  - Extract the `traversalTime` and calculate the mean speed (`segment_length / traversalTime`).
- Aggregate all mean speeds across all files (e.g., 85th percentile) to create a single global dictionary of `segment_free_flow_speed`.

**PASS 2: Iterative File Processing & Feature Engineering**
For every specified input file, execute the following:

**A. Data Standardization & Ghost Node Prep:**
- Parse wide-to-long and convert speeds to m/s.
- **Class Filter:** Drop "Motorcycles" ONLY. Retain Cars, Taxis, Medium/Heavy Vehicles, and Buses (Ghost Nodes for spatial occupancy).
- **Duration Filter:** Drop any vehicle whose total trajectory duration is < 5.0 seconds.
- **Temporal Downsampling:** Group by `track_id`, interpolate to exactly 1.0Hz.
- **Temporal Jitter:** Add uniform random noise (e.g., ± 0.2s) to `time`. Calculate `relative_time_gap` (min 0.1s).

**B. Map-Matching & Spatial Grounding:**
- Map-match to the UTM OSM network. Extract `segment_id`, `segment_length`, `segment_type`, `num_lanes`.
- Calculate `raw_offset`. Calculate `proportionate_distance_travelled = raw_offset / segment_length`.
- Derive parallel lane geometries to assign a discrete `lane_index`.
- Calculate `change_in_euclidean_distance` from the previous timestamp using UTM coordinates.
- Calculate the instantaneous heading/velocity vector for every vehicle to determine direction of travel.

**C. Asymmetric Ego-Centric Feature Engineering:**
- **Stratified Temporal CAV Sampling:** Calculate the entry time (`min(time)`) for every "Car" `track_id`. Divide the total timespan of the file into 5-minute (300-second) temporal bins. Group the `track_id`s by their entry bin and randomly sample exactly 10% of the IDs from each bin to ensure even temporal distribution. Label this combined sampled subset as CAVs (`is_CAV = True`).
- At every 1s timestamp, build a `scipy.spatial.KDTree` (UTM coordinates of ALL vehicles). Query using ONLY CAV coordinates for a 100m longitudinal environment.
- **CRITICAL TOPOLOGICAL & KINEMATIC FILTERS:** Before calculating zone averages, you MUST filter the KD-Tree neighbors using these 3 strict rules:
  1. **Segment Association:** Only include neighbors where `neighbor_segment_id == ego_segment_id`.
  2. **Direction:** Discard any neighbor moving in the opposite direction (e.g., heading difference > 90 degrees or negative dot product).
  3. **Lane Association:** - Proceeding/Following: Only include neighbors where `neighbor_lane_index == ego_lane_index`.
     - Leftwards/Rightwards: Only include neighbors on adjacent lanes (Left: `< ego_lane_index`, Right: `> ego_lane_index`).
- Project the filtered neighbors into the 4 zones: Proceeding (ahead max 50m), Following (behind max 50m), Leftwards (left lanes, 100m total), Rightwards (right lanes, 100m total).
- **Relative Occupancy:** - Proceeding/Following: `min(1.0, sum(neighbor_length + 2m gap) / 50m)`.
  - Left/Right: `min(1.0, sum(neighbor_length + 2m gap) / (100m * remaining_lane_count))`.
- **Speed & Density Averages:** Find average absolute speed (m/s) and raw vehicle counts (density) of the filtered vehicles per zone.

**D. Bounded Kinematic Normalization (Using Global Speeds):**
- Map the global `segment_free_flow_speed` from Pass 1 to the current segment.
- **Relative Kinematic Ratio:** `min(1.0, change_in_euclidean_distance / (segment_free_flow_speed * relative_time_gap))`.
- **Relative Speeds (Speed Indices):** - `relative_ego_speed` = ego_speed / segment_free_flow_speed
  - `relative_speed_[zone]` = average_zone_speed / segment_free_flow_speed
  - **CRITICAL IMPUTATION:** If a zone has no vehicles, set relative speed exactly to 1.0.

**E. Stream Optimization & Export:**
- Drop all rows where `is_CAV == False`. (Purge Ghost Nodes).
- Sort strictly chronologically by `time`, then by `track_id`.
- Export to a new CSV (e.g., `preprocessed_filename.csv`). Clear DataFrame from memory.

**Required Final Output Columns (Per File):**
`track_id`, `type`, `traveled_distance`, `avg_speed`, `lat`, `lon`, `speed`, `long_acc`, `lat_acc`, `time`, `segment_id`, `segment_length`, `segment_type`, `num_lanes`, `lane_index`, `proportionate_distance_travelled`, `change_in_euclidean_distance`, `relative_time_gap`, `relative_kinematic_ratio`, `relative_occupancy_proceeding`, `relative_occupancy_following`, `relative_occupancy_leftwards`, `relative_occupancy_rightwards`, `raw_density_proceeding`, `raw_density_following`, `raw_density_leftwards`, `raw_density_rightwards`, `segment_free_flow_speed`, `relative_ego_speed`, `relative_speed_proceeding`, `relative_speed_following`, `relative_speed_leftwards`, `relative_speed_rightwards`, `raw_speed_proceeding`, `raw_speed_following`, `raw_speed_leftwards`, `raw_speed_rightwards`.

Your job is to first create a plan to modify and expand the preprocess_pneuma.py script to achieve the above tasks. When you have come up with a robust plan, you will explain it step by step with no gaps. Then you can proceed to create the script in the current directory so that it can interact with the dataset stored in the pNEUMA_dataset folder. Only proceed to execute the plan after I have accepted it