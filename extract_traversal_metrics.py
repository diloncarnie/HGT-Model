import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import multiprocessing as mp
import argparse
import time
import json
from pathlib import Path
import warnings
import math
from functools import partial

# Suppress warnings if needed
warnings.filterwarnings('ignore')

def calculate_rtsm(temporal_speed, spatial_speed, temp_thresh, spat_thresh):
    if temp_thresh == -1.0 or spat_thresh == -1.0:
        return -1.0
        
    worst_case_distance = temp_thresh + spat_thresh
    if worst_case_distance == 0:
        return 0.0

    if temporal_speed == -1 or spatial_speed == -1:
        return 1.0

    if spatial_speed >= spat_thresh:
        if temporal_speed >= temp_thresh:
            distance = 0.0 # UpperRight
        else:
            distance = temp_thresh - temporal_speed # UpperLeft
    else:
        if not (temporal_speed >= temp_thresh):
            distance = (temp_thresh - temporal_speed) + (spat_thresh - spatial_speed) # LowerLeft
        else:
            distance = spat_thresh - spatial_speed # LowerRight

    rtsm = distance / worst_case_distance
    return float(max(0.0, min(1.0, rtsm)))

def process_track(track_data, config):
    """Processes all segments for a single track_id."""
    track_id, track_df = track_data
    track_df = track_df.sort_values('time').reset_index(drop=True)
    
    traversals = []
    segment_stats = {}
    
    # Identify segment transitions
    segment_changes = track_df['segment_id'] != track_df['segment_id'].shift()
    segment_groups = (segment_changes).cumsum()
    
    for group_id, group_df in track_df.groupby(segment_groups):
        if len(group_df) == 0:
            print(f"Track {track_id} has an empty segment group (group_id={group_id}). Skipping.")
            continue
            
        segment_id = group_df['segment_id'].iloc[0]
        if segment_id not in segment_stats:
            segment_stats[segment_id] = {'total': 0, 'invalid': 0}
        segment_stats[segment_id]['total'] += 1
        
        segment_length = group_df['segment_length'].iloc[0]
        
        # Gap filtering check to filter out single traversals
        time_diffs = group_df['time'].diff()
        if (time_diffs > config['gap_threshold']).any():
            segment_stats[segment_id]['invalid'] += 1
            continue
            
        # Monotonicity check
        valid_group = group_df[group_df['t_proj'] >= group_df['t_proj'].cummax()].copy()
        if len(valid_group) == 0:
            print(f"Track {track_id}, segment {segment_id} has no valid monotonic records. Skipping.")
            segment_stats[segment_id]['invalid'] += 1
            continue
            
        first_idx = valid_group.index[0]
        last_idx = valid_group.index[-1]
        
        records = []
        has_prev = False
        has_following = False
        
        # Previous Record
        if first_idx > 0:
            prev_row = track_df.loc[first_idx - 1].copy()
            if prev_row['segment_id'] != segment_id:
                prev_row['t_proj'] = -(prev_row['segment_length'] - prev_row['t_proj'])
                records.append(prev_row)
                has_prev = True
        else:
            if valid_group['t_proj'].iloc[0] <= max(config['min_edge_threshold'], min(config['max_edge_threshold'], config['edge_prop_threshold'] * segment_length)):
                has_prev = True
            
                
        # Current Records
        records.extend([row for _, row in valid_group.iterrows()])
        
        # Following Record
        if last_idx < len(track_df) - 1:
            next_row = track_df.loc[last_idx + 1].copy()
            if next_row['segment_id'] != segment_id:
                next_row['t_proj'] = segment_length + next_row['t_proj']
                records.append(next_row)
                has_following = True
        else:
            if segment_length - valid_group['t_proj'].iloc[-1] <= max(config['min_edge_threshold'], min(config['max_edge_threshold'], config['edge_prop_threshold'] * segment_length)):
                has_following = True
                
        # Quit if incomplete traversal (missing prev/next and not within threshold) or not enough records to interpolate
        if not has_prev or not has_following or len(records) < 2:
            if len(records) < 2:
                print(f"Track {track_id}, segment {segment_id} has insufficient records for interpolation (record_count={len(records)}). Skipping.")
            segment_stats[segment_id]['invalid'] += 1
            continue
        
        
            
        traversal_df = pd.DataFrame(records)
        
        # Adjust offsets: distanceOffsets[i] = max(offset, prev + 0.001)
        t_projs = traversal_df['t_proj'].values.copy()
        for i in range(1, len(t_projs)):
            t_projs[i] = max(t_projs[i], t_projs[i-1] + 0.001)
        traversal_df['t_proj'] = t_projs
        
        t_proj = traversal_df['t_proj'].values
        times = traversal_df['time'].values
        speeds = traversal_df['speed'].values
        
        # Interpolate Time-Distance and Speed-Distance
        try:
            f_time = interp1d(t_proj, times, kind='linear', bounds_error=False, fill_value="extrapolate")
            f_speed = interp1d(t_proj, speeds, kind='linear', bounds_error=False, fill_value="extrapolate")
            
            # Calculate traversal length
            length = segment_length
                
            traversal_time = float(f_time(length) - f_time(0.0))
            if traversal_time <= 0:
                temporal_mean_speed = valid_group['speed'].mean()
            else:
                temporal_mean_speed = length / traversal_time
                
            # Compute Spatial Mean Speed
            first_offset = max(0.0, t_proj[0])
            last_offset = min(length, t_proj[-1])
            cur = math.ceil(first_offset)
            end = math.floor(last_offset)
            
            if end - cur < config['speed_sample_interval']:
                spatial_mean_speed = valid_group['speed'].mean()
            else:
                eval_points = []
                temp_cur = float(cur)
                while end - temp_cur >= config['speed_sample_interval']:
                    eval_points.append(temp_cur)
                    temp_cur += config['speed_sample_interval']
                spatial_mean_speed = np.mean(f_speed(np.array(eval_points)))
                
        except Exception as e:
            # Fallback on error
            print(f"Interpolation error for track {track_id}, segment {segment_id}: {e}")
            total_dist = t_proj[-1] - t_proj[0]
            total_t = times[-1] - times[0]
            if total_t > 0:
                temporal_mean_speed = total_dist / total_t
            else:
                temporal_mean_speed = valid_group['speed'].mean()
            spatial_mean_speed = valid_group['speed'].mean()
            traversal_time = total_t

        # Calculate stopping duration
        vg_times = valid_group['time'].values
        vg_speeds = valid_group['speed'].values
        stopping_duration = 0.0
        for i in range(1, len(valid_group)):
            if vg_speeds[i] < config['stopped_speed_threshold'] and vg_speeds[i-1] < config['stopped_speed_threshold']:
                stopping_duration += vg_times[i] - vg_times[i-1]

        traversals.append({
            'segment_id': segment_id,
            'track_id': track_id,
            'timestamp': valid_group['time'].iloc[-1],
            'temporal_mean_speed': temporal_mean_speed,
            'spatial_mean_speed': spatial_mean_speed,
            'stopping_duration': stopping_duration,
            'traversal_time': traversal_time,
            'segment_length': segment_length
        })
        
    return traversals, segment_stats

def process_file(csv_path, config):
    t0 = time.time()
    print(f"Processing: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return 0.0

    print(f"Loaded {len(df)} rows. Time: {time.time() - t0:.2f}s")
    t1 = time.time()
    
    # Process by track_id to allow matching previous/following records correctly
    grouped = list(df.groupby('track_id'))
    print(f"Grouped into {len(grouped)} tracks. Time: {time.time() - t1:.2f}s")
    
    t2 = time.time()
    valid_traversals = []
    global_segment_stats = {}
    
    process_func = partial(process_track, config=config)
    with mp.Pool() as pool:
        results = pool.map(process_func, grouped)
        
    for res_traversals, res_stats in results:
        valid_traversals.extend(res_traversals)
        for sid, stats in res_stats.items():
            if sid not in global_segment_stats:
                global_segment_stats[sid] = {'total': 0, 'invalid': 0}
            global_segment_stats[sid]['total'] += stats['total']
            global_segment_stats[sid]['invalid'] += stats['invalid']
            
    print(f"Spline fitting and traversal extraction took {time.time() - t2:.2f}s")
    print(f"Extracted {len(valid_traversals)} valid traversals.")
    
    if not valid_traversals:
        print("No valid traversals found.")
        return 0.0

    traversals_df = pd.DataFrame(valid_traversals)
    
    t3 = time.time()
    # Dynamic Segment Thresholds matching Java EXACTLY
    segment_thresholds = {}
    file_max_red_light_duration = 0.0
    file_max_red_light_segment_id = None
    
    for segment_id, seg_df in traversals_df.groupby('segment_id'):
        seg_length = seg_df['segment_length'].iloc[0]
        
        times = seg_df['traversal_time'].values
        
        # Temporal Threshold
        temp_thresh = None
        red_light_duration = 0.0
        # Calculate threshold if there is more than one traversal
        if len(times) > 1:
            filtered_times = [t for t in times if t > config['min_traversal_time']]
            if len(filtered_times) > 0:
                p5_time = np.percentile(times, 5)
                red_light_duration = np.percentile(seg_df['stopping_duration'], 95)
                if red_light_duration > file_max_red_light_duration:
                    file_max_red_light_duration = red_light_duration
                    file_max_red_light_segment_id = segment_id
                ideal_time = p5_time + red_light_duration
                if ideal_time > 0:
                    temp_thresh = seg_length / ideal_time
                else:
                    print(f"Segment {segment_id} has non-positive ideal time for temporal threshold calculation (p5_time={p5_time}, red_light_duration={red_light_duration})")
            else:
                print(f"Segment {segment_id} has no traversals longer than 2.0 seconds for temporal threshold calculation (count={len(times)}, length={seg_length})")
        else:
            print(f"Segment {segment_id} has insufficient traversals for temporal threshold calculation (count={len(times)}, length={seg_length})")
                    
        # Spatial Threshold
        spat_thresh = None
        free_flow_speed = -1.0
        if len(seg_df) > 1 and temp_thresh is not None:
            good_temp_df = seg_df[seg_df['temporal_mean_speed'] >= temp_thresh]
            if len(good_temp_df) > 0:
                spat_thresh = np.percentile(good_temp_df['spatial_mean_speed'], 5)
                free_flow_speed = np.percentile(good_temp_df['spatial_mean_speed'], 85)
            else:
                print(f"Segment {segment_id} has no traversals meeting temporal threshold for spatial threshold calculation (count={len(good_temp_df)}, temp_thresh={temp_thresh}, length={seg_length})")
        else:
            print(f"Segment {segment_id} has insufficient data for spatial threshold calculation (count={len(seg_df)}, temp_thresh={temp_thresh}, lenght={seg_length})")
                
        if temp_thresh is None or spat_thresh is None:
            print(f"Segment {segment_id} has invalid thresholds. Total={global_segment_stats[segment_id]['total']} Invalid={global_segment_stats[segment_id]['invalid']}.  Adding with default thresholds.")
            temp_thresh = -1.0
            spat_thresh = -1.0
            
        segment_thresholds[int(segment_id)] = {
            'temporal_threshold': temp_thresh,
            'spatial_threshold': spat_thresh,
            'free_flow_speed': free_flow_speed,
            'red_light_duration': red_light_duration,
            'total_traversals': global_segment_stats[segment_id]['total'],
            'invalid_traversals': global_segment_stats[segment_id]['invalid']
        }

    # Add segments that only had invalid traversals
    for segment_id, stats in global_segment_stats.items():
        
        sid_int = int(segment_id)
        if sid_int not in segment_thresholds:
            print(f"Segment {segment_id} has no valid traversals. Total={stats['total']}  Invalid={stats['invalid']}. Adding with default thresholds.")
            segment_thresholds[sid_int] = {
                'temporal_threshold': -1.0,
                'spatial_threshold': -1.0,
                'free_flow_speed': -1.0,
                'red_light_duration': 0.0,
                'total_traversals': stats['total'],
                'invalid_traversals': stats['invalid']
            }

    print(f"Threshold calculation took {time.time() - t3:.2f}s")
    
    t4 = time.time()
    # RTSM Calculation
    rtsm_values = []
    for _, row in traversals_df.iterrows():
        sid = int(row['segment_id'])
        t_thresh = segment_thresholds[sid]['temporal_threshold']
        s_thresh = segment_thresholds[sid]['spatial_threshold']
        rtsm = calculate_rtsm(row['temporal_mean_speed'], row['spatial_mean_speed'], t_thresh, s_thresh)
        rtsm_values.append(rtsm)
        
    traversals_df['rtsm'] = rtsm_values
    print(f"RTSM Calculation took {time.time() - t4:.2f}s")
    
    out_dir = Path(csv_path).parent
    
    with open(out_dir / 'segment_thresholds.json', 'w') as f:
        json.dump(segment_thresholds, f, indent=4)
        
    out_cols = ['segment_id', 'track_id', 'timestamp', 'temporal_mean_speed', 'spatial_mean_speed', 'rtsm']
    traversals_df.sort_values('segment_id')[out_cols].to_csv(out_dir / 'traversal_metrics.csv', index=False)
        
    print(f"Total time for {csv_path}: {time.time() - t0:.2f}s\n")
    
    if file_max_red_light_duration > 0:
        print(f"Max red light duration for this file: {file_max_red_light_duration:.2f}s (Segment ID: {file_max_red_light_segment_id})\n")
        
    return file_max_red_light_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vehicle traversal metrics from matched trajectories.")
    parser.add_argument("--file", type=str, help="Path to a single matched_trajectories.csv file")
    parser.add_argument("--folder", type=str, help="Root directory to search recursively for matched_trajectories.csv")
    
    args = parser.parse_args()
    
    config = {
        "gap_threshold": 30.0,
        "min_edge_threshold": 8.0,
        "max_edge_threshold": 25.0,
        "edge_prop_threshold": 0.3,
        "connection_length_threshold": 5.0,
        "speed_sample_interval": 10.0,
        "stopped_speed_threshold": 0.1,
        "min_traversal_time": 2.0
    }
    
    max_red_light_durations = []

    if args.file:
        res = process_file(args.file, config)
        if res > 0:
            max_red_light_durations.append(res)
    elif args.folder:
        root_path = Path(args.folder)
        for csv_path in root_path.rglob("matched_trajectories_filtered.csv"):
            res = process_file(csv_path, config)
            if res > 0:
                max_red_light_durations.append(res)
    else:
        print("Please provide either --file or --folder")
        
    if len(max_red_light_durations) > 1:
        avg_max_red_light = sum(max_red_light_durations) / len(max_red_light_durations)
        print(f"Average of max red light durations across {len(max_red_light_durations)} files: {avg_max_red_light:.2f}s")