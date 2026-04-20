import pandas as pd
import numpy as np
import argparse
import time
import json
from pathlib import Path

def calculate_rtsm(temporal_speed, spatial_speed, temp_thresh, spat_thresh):
    worst_case_distance = temp_thresh + spat_thresh
    if worst_case_distance == 0:
        return 0.0

    if temporal_speed >= temp_thresh and spatial_speed >= spat_thresh:
        distance = 0.0
    elif spatial_speed >= spat_thresh and temporal_speed < temp_thresh:
        distance = temp_thresh - temporal_speed
    elif temporal_speed < temp_thresh and spatial_speed < spat_thresh:
        distance = (temp_thresh - temporal_speed) + (spat_thresh - spatial_speed)
    else:
        distance = spat_thresh - spatial_speed

    rtsm = distance / worst_case_distance
    return max(0.0, min(1.0, rtsm))

def compute_tema_for_segment(segment_id, seg_df, segment_thresholds, tau=30.0):
    # Sort strictly chronologically
    seg_df = seg_df.sort_values('timestamp').reset_index(drop=True)
    
    if len(seg_df) == 0:
        return pd.DataFrame()
        
    thresh = segment_thresholds.get(str(segment_id))
    if not thresh:
        # Fallback if no thresholds found
        temp_thresh, spat_thresh = 0.0, 0.0
    else:
        temp_thresh = thresh['temporal_threshold']
        spat_thresh = thresh['spatial_threshold']
        
    s_temp = temp_thresh
    s_spat = spat_thresh
    
    ema_temp = []
    ema_spat = []
    recalc_rtsm = []
    time_since_last_update = []
    
    last_timestamp = None
    
    for i, row in seg_df.iterrows():
        current_ts = row['timestamp']
        if last_timestamp is None:
            dt = 300.0
        else:
            dt = current_ts - last_timestamp
            
        if dt < 0:
            dt = 0.0 # Should not happen due to sort, but just in case
            
        time_since_last_update.append(dt)
        
        alpha = 1.0 - np.exp(-dt / tau)
        
        x_temp = row['temporal_mean_speed']
        x_spat = row['spatial_mean_speed']
        
        s_temp = (alpha * x_temp) + ((1.0 - alpha) * s_temp)
        s_spat = (alpha * x_spat) + ((1.0 - alpha) * s_spat)
        
        ema_temp.append(s_temp)
        ema_spat.append(s_spat)
        
        rtsm = calculate_rtsm(s_temp, s_spat, temp_thresh, spat_thresh)
        recalc_rtsm.append(rtsm)
        
        last_timestamp = current_ts
        
    seg_df['ema_temporal_speed'] = ema_temp
    seg_df['ema_spatial_speed'] = ema_spat
    seg_df['recalculated_rtsm'] = recalc_rtsm
    seg_df['time_since_last_update'] = time_since_last_update
    
    out_cols = ['segment_id', 'timestamp', 'ema_temporal_speed', 'ema_spatial_speed', 'recalculated_rtsm', 'time_since_last_update']
    return seg_df[out_cols]

def downsample_states(states_df, interval=30.0):
    if len(states_df) == 0:
        return pd.DataFrame()
        
    downsampled = []
    for segment_id, seg_df in states_df.groupby('segment_id'):
        seg_df = seg_df.sort_values('timestamp').reset_index(drop=True)
        last_kept_ts = None
        
        for i, row in seg_df.iterrows():
            if last_kept_ts is None or (row['timestamp'] - last_kept_ts) >= interval:
                downsampled.append(row)
                last_kept_ts = row['timestamp']
                
    if not downsampled:
        return pd.DataFrame(columns=states_df.columns)
        
    return pd.DataFrame(downsampled)

def process_file(csv_path, tau=30.0, downsample_interval=30.0):
    t0 = time.time()
    out_dir = Path(csv_path).parent
    thresh_path = out_dir / 'segment_thresholds.json'
    
    if not thresh_path.exists():
        print(f"Warning: {thresh_path} not found. Skipping {csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path)
        with open(thresh_path, 'r') as f:
            segment_thresholds = json.load(f)
    except Exception as e:
        print(f"Error reading data in {out_dir}: {e}")
        return
        
    print(f"Loaded {len(df)} rows from {csv_path}. Time: {time.time() - t0:.2f}s")
    
    t1 = time.time()
    # Compute Continuous-Time EMA per segment
    continuous_dfs = []
    for segment_id, seg_df in df.groupby('segment_id'):
        continuous_seg_df = compute_tema_for_segment(segment_id, seg_df, segment_thresholds, tau)
        if not continuous_seg_df.empty:
            continuous_dfs.append(continuous_seg_df)
            
    if not continuous_dfs:
        print("No continuous states generated.")
        return
        
    continuous_states = pd.concat(continuous_dfs, ignore_index=True)
    print(f"T-EMA calculation took {time.time() - t1:.2f}s")
    
    t2 = time.time()
    # Minimum-Gap Downsampling
    downsampled_states = downsample_states(continuous_states, interval=downsample_interval)
    print(f"Downsampling took {time.time() - t2:.2f}s")
    
    # Export
    continuous_states.to_csv(out_dir / 'continuous_aggregated_states.csv', index=False)
    downsampled_states.to_csv(out_dir / 'downsampled_aggregated_states.csv', index=False)
    print(f"Total time for {csv_path}: {time.time() - t0:.2f}s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate traffic states using T-EMA and downsampling.")
    parser.add_argument("--file", type=str, help="Path to a single traversal_metrics.csv file")
    parser.add_argument("--folder", type=str, help="Root directory to search recursively for traversal_metrics.csv")
    parser.add_argument("--tau", type=float, default=30.0, help="Time constant tau for T-EMA (seconds)")
    parser.add_argument("--interval", type=float, default=10.0, help="Minimum gap interval for downsampling (seconds)")
    
    args = parser.parse_args()
    
    if args.file:
        process_file(args.file, args.tau, args.interval)
    elif args.folder:
        root_path = Path(args.folder)
        for csv_path in root_path.rglob("traversal_metrics.csv"):
            process_file(csv_path, args.tau, args.interval)
    else:
        print("Please provide either --file or --folder")