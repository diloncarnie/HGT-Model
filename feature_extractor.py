import argparse
import time
import os
import json
import multiprocessing
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def process_frenet_continuous_chunk(chunk_data_tuple):
    ego_df, bg_df, adjacency, speeds, config = chunk_data_tuple
    
    if ego_df.empty or bg_df.empty:
        return pd.DataFrame()
        
    # Build KDTree for all background points in this time chunk
    coords = bg_df[['x', 'y']].values
    tree = KDTree(coords)
    
    results = []
    time_thresh = config.get("frenet_time_diff_thresh", 1.1)
    
    # Optimization: Extract arrays to avoid pandas overhead in the inner loop
    bg_track_ids = bg_df['track_id'].values
    bg_times = bg_df['time'].values
    
    for idx, ego in ego_df.iterrows():
        t_ego = ego['time']
        ego_pos = np.array([ego['x'], ego['y']])
        ego_seg = ego['segment_id']
        ego_D = ego['D']
        ego_S = ego['t_proj']
        ego_az = ego['azcar']
        ego_track_id = ego['track_id']
        
        # 1. Spatial filter
        neighbors_idx = tree.query_ball_point(ego_pos, r=config["kdtree_radius"])
        
        valid_neighbors = []
        if len(neighbors_idx) > 0:
            n_idx_arr = np.array(neighbors_idx)
            
            # Filter out the ego vehicle itself
            mask_not_ego = bg_track_ids[n_idx_arr] != ego_track_id
            n_idx_arr = n_idx_arr[mask_not_ego]
            
            if len(n_idx_arr) > 0:
                n_times = bg_times[n_idx_arr]
                n_tracks = bg_track_ids[n_idx_arr]
                
                # 2. Time filter (Vectorized)
                time_diffs = np.abs(n_times - t_ego)
                valid_time_mask = time_diffs <= time_thresh
                
                n_idx_arr = n_idx_arr[valid_time_mask]
                n_tracks = n_tracks[valid_time_mask]
                time_diffs = time_diffs[valid_time_mask]
                
                if len(n_idx_arr) > 0:
                    # 3. Closest point in time per vehicle (Highly Vectorized)
                    temp_df = pd.DataFrame({'idx': n_idx_arr, 'td': time_diffs, 'tid': n_tracks})
                    best_indices = temp_df.loc[temp_df.groupby('tid')['td'].idxmin(), 'idx']
                    
                    n_df = bg_df.iloc[best_indices]
                    
                    ego_adj = adjacency.get(ego_seg, {})
                    successors = ego_adj.get('successors', [])
                    predecessors = ego_adj.get('predecessors', [])
                    successor_lengths = ego_adj.get('successor_lengths', [])
                    predecessor_lengths = ego_adj.get('predecessor_lengths', [])
                    
                    for _, n_row in n_df.iterrows():
                        n_seg = n_row['segment_id']
                        n_az = n_row['azcar']
                        
                        heading_diff = abs((n_az - ego_az + 180) % 360 - 180)
                        if heading_diff > config["frenet_heading_diff_thresh"]:
                            continue
                            
                        # Topological Check
                        if n_seg == ego_seg:
                            delta_S = n_row['t_proj'] - ego_S
                        elif n_seg in successors:
                            idx_s = successors.index(n_seg)
                            intermediate_dist = sum(successor_lengths[:idx_s])
                            delta_S = (ego['segment_length'] - ego_S) + intermediate_dist + n_row['t_proj']
                        elif n_seg in predecessors:
                            idx_p = predecessors.index(n_seg)
                            intermediate_dist = sum(predecessor_lengths[:idx_p])
                            delta_S = -ego_S - intermediate_dist - (n_row['segment_length'] - n_row['t_proj'])
                        else:
                            continue
                            
                        delta_D = n_row['D'] - ego_D
                        valid_neighbors.append({
                            'delta_S': delta_S,
                            'delta_D': delta_D,
                            'speed': n_row['speed'],
                            'type': n_row['type']
                        })
            
        # 6 Zones: Proceeding, Following, Left_Proc, Left_Foll, Right_Proc, Right_Foll
        zones = {
            'proceeding': [], 'following': [], 
            'leftwards_proceeding': [], 'leftwards_following': [],
            'rightwards_proceeding': [], 'rightwards_following': []
        }
        
        for n in valid_neighbors:
            dS = n['delta_S']
            dD = n['delta_D']
            v_len = config["frenet_v_len_car"] if n['type'] in ['Car', 'Taxi'] else config["frenet_v_len_heavy"]
            
            if -config["frenet_delta_s_thresh"] <= dS <= config["frenet_delta_s_thresh"]:
                if -config["frenet_delta_d_thresh"] <= dD <= config["frenet_delta_d_thresh"]:
                    if dS > 0: zones['proceeding'].append((n['speed'], v_len))
                    elif dS < 0: zones['following'].append((n['speed'], v_len))
                elif dD > config["frenet_delta_d_thresh"]:
                    if dS >= 0: zones['leftwards_proceeding'].append((n['speed'], v_len))
                    else: zones['leftwards_following'].append((n['speed'], v_len))
                elif dD < -config["frenet_delta_d_thresh"]:
                    if dS >= 0: zones['rightwards_proceeding'].append((n['speed'], v_len))
                    else: zones['rightwards_following'].append((n['speed'], v_len))
                
        ego_res = ego.to_dict()
        rem_lanes = max(1, ego['num_lanes'] - 1)
        
        for z in zones.keys():
            if not zones[z]:
                ego_res[f'raw_density_{z}'] = 0
                ego_res[f'raw_speed_{z}'] = speeds.get(ego_seg, config["default_speed_fallback"]) 
                ego_res[f'relative_occupancy_{z}'] = 0.0
                ego_res[f'relative_speed_{z}'] = 1.0
            else:
                ego_res[f'raw_density_{z}'] = len(zones[z])
                ego_res[f'raw_speed_{z}'] = np.mean([x[0] for x in zones[z]])
                
                # Occupancy = sum(length + 2) / Area
                occ_sum = sum([x[1] + config["frenet_occ_buffer"] for x in zones[z]])
                if z in ['proceeding', 'following']:
                    occ = min(1.0, occ_sum / config["frenet_delta_s_thresh"])
                else:
                    # Side zones are 50m long (approx) and rem_lanes wide
                    occ = min(1.0, occ_sum / (config["frenet_delta_s_thresh"] * rem_lanes))
                ego_res[f'relative_occupancy_{z}'] = occ
                
                ego_res[f'relative_speed_{z}'] = ego_res[f'raw_speed_{z}'] / max(1.0, speeds.get(ego_seg, config["default_speed_fallback"]))
                
        results.append(ego_res)
        
    return pd.DataFrame(results)

def process_single_folder(folder_path, config):
    print(f"\n--- Processing Folder: {folder_path} ---")
    folder_start_time = time.time()
    
    matched_file = os.path.join(folder_path, "matched_trajectories_filtered.csv")
    if not os.path.exists(matched_file):
        matched_file = os.path.join(folder_path, "matched_trajectories.csv")
        
    speeds_file = os.path.join(folder_path, "empirical_free_flow_speeds.json")
    
    if not os.path.exists(matched_file):
        print(f"Skipping {folder_path}: missing matched_trajectories_filtered.csv or matched_trajectories.csv")
        return
    if not os.path.exists(speeds_file):
        print(f"Skipping {folder_path}: missing required empirical_free_flow_speeds.json")
        return
        
    df = pd.read_csv(matched_file, dtype={'track_id': str, 'segment_id': str})
    if df.empty:
        print("Matched trajectories file is empty.")
        return
        
    with open(speeds_file, 'r') as f:
        empirical_speeds = json.load(f)
        
    adjacency = config["topological_adjacency"]
    
    # ---------------------------------------------------------
    # PIPELINE INTEGRATION: Spatial & Ego-Centric Extraction
    # ---------------------------------------------------------
    print("\n--- Starting Ego-Centric Pipeline ---")
    
    # Prepare DataFrame columns for Ego-Centric extraction
    df['segment_type'] = df['highway']
    df.rename(columns={
        'traveled_d': 'traveled_distance', 
        'lon_acc': 'long_acc', 
        'prop_dist': 'proportionate_distance_travelled'
    }, inplace=True)

    # Sampling 15% of all unique Car/Taxi tracks as CAVs
    print("Sampling 15% of unique Car/Taxi tracks as CAVs...")
    df['is_CAV'] = False
    unique_tracks = df[df['type'].isin(['Car', 'Taxi'])]['track_id'].unique()
    n_sample = max(1, int(len(unique_tracks) * config["cav_sample_percentage"]))
    sampled_cavs = np.random.choice(unique_tracks, n_sample, replace=False)
    df.loc[df['track_id'].isin(sampled_cavs), 'is_CAV'] = True
    print(f"Total CAVs selected: {len(sampled_cavs)} out of {len(unique_tracks)} unique Car/Taxi vehicles.")
        
    # Frenet Ego-Centric Extraction
    print("Extracting spatial features via KDTree (using continuous time-window chunking)...")
    kdtree_start = time.time()
    
    time_groups = []
    chunk_duration = 10.0 # process 10 seconds of ego data at a time
    min_time = df['time'].min()
    max_time = df['time'].max()
    time_thresh = config.get("frenet_time_diff_thresh", 1.1)
    
    t_start = min_time
    while t_start <= max_time:
        t_end = t_start + chunk_duration
        
        ego_df = df[(df['is_CAV'] == True) & (df['time'] >= t_start) & (df['time'] < t_end)]
        if not ego_df.empty:
            bg_df = df[(df['time'] >= t_start - time_thresh) & (df['time'] <= t_end + time_thresh)]
            time_groups.append((ego_df.copy(), bg_df.copy(), adjacency, empirical_speeds, config))
            
        t_start = t_end
        
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(process_frenet_continuous_chunk, time_groups)
    pool.close()
    pool.join()
    
    valid_results = [r for r in results if not r.empty]
    final_df = pd.concat(valid_results, ignore_index=True) if valid_results else pd.DataFrame()
    print(f"Feature extraction took {time.time() - kdtree_start:.2f} seconds")
    
    if final_df.empty:
        print("No valid CAV features extracted. Exiting.")
        return

    # Deferred Kinematics Calculation (Calculated only for remaining CAVs)
    print("Calculating final kinematics...")
    kin_start = time.time()
    
    final_df['segment_free_flow_speed'] = final_df['segment_id'].map(empirical_speeds).fillna(config["default_speed_fallback"])
    final_df['relative_ego_speed'] = final_df['speed'] / final_df['segment_free_flow_speed']
    
    final_df = final_df.sort_values(['track_id', 'time'])
    
    final_df['change_in_euclidean_distance'] = final_df.groupby('track_id').apply(
        lambda x: np.sqrt((x['x'].diff()**2) + (x['y'].diff()**2))
    ).reset_index(level=0, drop=True)
    
    final_df['relative_time_gap'] = final_df.groupby('track_id')['time'].diff()
    
    final_df['relative_kinematic_ratio'] = final_df['change_in_euclidean_distance'] / (final_df['segment_free_flow_speed'] * final_df['relative_time_gap'])
    final_df['relative_kinematic_ratio'] = final_df['relative_kinematic_ratio'].clip(upper=1.0)
    
    print(f"Kinematics took {time.time() - kin_start:.2f} seconds")
    
    # Export Final Fully Processed CSV
    out_cols = [
        'track_id', 'type', 'traveled_distance', 'avg_speed', 'lat', 'lon', 'speed', 'long_acc', 'lat_acc', 'time',
        'segment_id', 'segment_length', 'segment_type', 'num_lanes', 'lane_index', 'proportionate_distance_travelled',
        'change_in_euclidean_distance', 'relative_time_gap', 'relative_kinematic_ratio',
        'segment_free_flow_speed', 'relative_ego_speed', 'is_outlier'
    ]
    
    for z in ['proceeding', 'following', 'leftwards_proceeding', 'leftwards_following', 'rightwards_proceeding', 'rightwards_following']:
        out_cols.extend([f'relative_occupancy_{z}', f'raw_density_{z}', f'relative_speed_{z}', f'raw_speed_{z}'])
    
    folder_name = os.path.basename(folder_path)
    final_out_path = os.path.join(folder_path, f"{folder_name}_processed.csv")
    
    final_df[out_cols].to_csv(final_out_path, index=False)
    
    print(f"\nFinished processing! Saved to {final_out_path}")
    print(f"Execution for this folder took {time.time() - folder_start_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Path to processing directory or a specific folder")
    args = parser.parse_args()
    
    overall_start = time.time()
    
    print("Loading topological adjacency...")
    if not os.path.exists('topological_adjacency_merged.json'):
        print("Error: topological_adjacency_merged.json not found. Make sure to run the external OSM downloader script.")
        return
        
    with open('topological_adjacency_merged.json', 'r') as f:
        topological_adjacency = json.load(f)

    config = {
        "cav_sample_percentage": 0.15,
        "kdtree_radius": 55,
        "frenet_heading_diff_thresh": 65.0,
        "frenet_time_diff_thresh": 1.1,
        "frenet_delta_s_thresh": 50.0,
        "frenet_delta_d_thresh": 1.6,
        "frenet_v_len_car": 5.0,
        "frenet_v_len_heavy": 12.5,
        "frenet_occ_buffer": 2.0,
        "default_speed_fallback": 10.0,
        "topological_adjacency": topological_adjacency
    }
    
    # Discover target folders
    if os.path.isdir(args.input_path):
        # Is it a specific folder?
        if os.path.exists(os.path.join(args.input_path, 'matched_trajectories_filtered.csv')) or \
           os.path.exists(os.path.join(args.input_path, 'matched_trajectories.csv')):
            folders = [args.input_path]
        else:
            folders = []
            for root, dirs, files in os.walk(args.input_path):
                if 'matched_trajectories_filtered.csv' in files or 'matched_trajectories.csv' in files:
                    folders.append(root)
    else:
        print(f"Error: {args.input_path} is not a valid directory.")
        return
        
    folders = sorted(list(set(folders)))
    
    if not folders:
        print("No folders with matched trajectories found.")
        return
        
    for folder in folders:
        process_single_folder(folder, config)

    print(f"\nTotal feature extraction execution completed in {time.time() - overall_start:.2f} seconds.")

if __name__ == '__main__':
    main()