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
import logging

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
    
    # Keep original index to update outliers later
    track_df['original_index'] = track_df.index
    track_df = track_df.sort_values('time').reset_index(drop=True)
    
    traversals = []
    segment_stats = {}
    logs = []
    valid_original_indices = []
    
    # Identify segment transitions
    segment_changes = track_df['segment_id'] != track_df['segment_id'].shift()
    segment_groups = (segment_changes).cumsum()
    
    for group_id, group_df in track_df.groupby(segment_groups):
        if len(group_df) == 0:
            logs.append(f"Track {track_id} has an empty segment group (group_id={group_id}). Skipping.")
            continue
            
        segment_id = group_df['segment_id'].iloc[0]
        if segment_id not in segment_stats:
            segment_stats[segment_id] = {
                'total': 0, 'invalid': 0, 'invalid_outlier_ratio': 0, 
                'invalid_outlier_stop': 0, 'invalid_gap': 0, 
                'invalid_monotonic': 0, 'invalid_bounds': 0, 'insufficient_records': 0,
                'sum_outlier_ratio': 0.0, 'count_outlier_ratio': 0,
                'sum_outlier_stop': 0.0, 'count_outlier_stop': 0
            }
        segment_stats[segment_id]['total'] += 1
        
        segment_length = group_df['segment_length'].iloc[0]
        
        # Outlier Traversal Check: Proportion-based & Kinematic-Gated rejection
        if 'is_outlier' in group_df.columns:
            outlier_ratio = group_df['is_outlier'].mean()
            segment_stats[segment_id]['sum_outlier_ratio'] += float(outlier_ratio)
            segment_stats[segment_id]['count_outlier_ratio'] += 1
            
            outlier_stopped_mask = group_df['is_outlier'] & (group_df['speed'] < config['stopped_speed_threshold'])
            max_outlier_stop = 0.0
            if outlier_stopped_mask.any():
                current_outlier_stop = 0.0
                times = group_df['time'].values
                mask_vals = outlier_stopped_mask.values
                
                for i in range(1, len(group_df)):
                    if mask_vals[i] and mask_vals[i-1]:
                        current_outlier_stop += times[i] - times[i-1]
                    else:
                        max_outlier_stop = max(max_outlier_stop, current_outlier_stop)
                        current_outlier_stop = 0.0
                max_outlier_stop = max(max_outlier_stop, current_outlier_stop)
                
                segment_stats[segment_id]['sum_outlier_stop'] += float(max_outlier_stop)
                segment_stats[segment_id]['count_outlier_stop'] += 1
                    
            if outlier_ratio > config.get('max_outlier_proportion', 0.2):
                segment_stats[segment_id]['invalid'] += 1
                segment_stats[segment_id]['invalid_outlier_ratio'] += 1
                continue
                
            if max_outlier_stop > config.get('max_outlier_stop_duration', 5.0):
                segment_stats[segment_id]['invalid'] += 1
                segment_stats[segment_id]['invalid_outlier_stop'] += 1
                continue
        
        # Gap filtering check to filter out single traversals
        time_diffs = group_df['time'].diff()
        if (time_diffs > config['gap_threshold']).any():
            segment_stats[segment_id]['invalid'] += 1
            segment_stats[segment_id]['invalid_gap'] += 1
            continue
            
        # Monotonicity check
        valid_group = group_df[group_df['t_proj'] >= group_df['t_proj'].cummax()].copy()
        if len(valid_group) == 0:
            logs.append(f"Track {track_id}, segment {segment_id} has no valid monotonic records. Skipping.")
            segment_stats[segment_id]['invalid'] += 1
            segment_stats[segment_id]['invalid_monotonic'] += 1
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
            segment_stats[segment_id]['invalid'] += 1
            if len(records) < 2:
                segment_stats[segment_id]['insufficient_records'] += 1
            else:
                segment_stats[segment_id]['invalid_bounds'] += 1
            continue
        
        
            
        traversal_df = pd.DataFrame(records)
        
        # Adjust offsets: distanceOffsets[i] = max(offset, prev + config['monotonicity_offset'])
        t_projs = traversal_df['t_proj'].values.copy()
        for i in range(1, len(t_projs)):
            t_projs[i] = max(t_projs[i], t_projs[i-1] + config['monotonicity_offset'])
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
            logs.append(f"Interpolation error for track {track_id}, segment {segment_id}: {e}")
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
        max_stopping_duration = 0.0
        current_stop_duration = 0.0
        for i in range(1, len(valid_group)):
            if vg_speeds[i] < config['stopped_speed_threshold'] and vg_speeds[i-1] < config['stopped_speed_threshold']:
                current_stop_duration += vg_times[i] - vg_times[i-1]
            else:
                if current_stop_duration > max_stopping_duration:
                    max_stopping_duration = current_stop_duration
                current_stop_duration = 0.0
                
        if current_stop_duration > max_stopping_duration:
            max_stopping_duration = current_stop_duration
            
        stopping_duration = max_stopping_duration

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
        valid_original_indices.extend(group_df['original_index'].tolist())
        
    return traversals, segment_stats, logs, valid_original_indices

def process_file(csv_path, config):
    out_dir = Path(csv_path).parent
    log_path = out_dir / 'traversal-metrics.log'
    
    # Setup standard python logging
    logger = logging.getLogger(f"extraction_{csv_path}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate lines if function runs multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
        
    logger.info(f"--- Extraction Log for {csv_path} ---")
    
    t0 = time.time()
    logger.info(f"Processing: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}")
        return 0.0, 0.0, 0.0, [], []

    logger.info(f"Loaded {len(df)} rows. Time: {time.time() - t0:.2f}s")
    t1 = time.time()
    
    # Process by track_id to allow matching previous/following records correctly
    grouped = df.groupby('track_id')
    logger.info(f"Grouped into {grouped.ngroups} tracks. Time: {time.time() - t1:.2f}s")
    
    t2 = time.time()
    valid_traversals = []
    all_valid_indices = []
    global_segment_stats = {}
    
    process_func = partial(process_track, config=config)
    with mp.Pool() as pool:
        for res_traversals, res_stats, res_logs, res_indices in pool.imap_unordered(process_func, grouped):
            for msg in res_logs:
                logger.info(msg)
            valid_traversals.extend(res_traversals)
            all_valid_indices.extend(res_indices)
            for sid, stats in res_stats.items():
                if sid not in global_segment_stats:
                    global_segment_stats[sid] = {
                        'total': 0, 'invalid': 0, 'invalid_outlier_ratio': 0, 
                        'invalid_outlier_stop': 0, 'invalid_gap': 0,
                        'invalid_monotonic': 0, 'invalid_bounds': 0, 'insufficient_records': 0,
                        'sum_outlier_ratio': 0.0, 'count_outlier_ratio': 0,
                        'sum_outlier_stop': 0.0, 'count_outlier_stop': 0
                    }
                global_segment_stats[sid]['total'] += stats['total']
                global_segment_stats[sid]['invalid'] += stats['invalid']
                global_segment_stats[sid]['invalid_outlier_ratio'] += stats['invalid_outlier_ratio']
                global_segment_stats[sid]['invalid_outlier_stop'] += stats['invalid_outlier_stop']
                global_segment_stats[sid]['invalid_gap'] += stats['invalid_gap']
                global_segment_stats[sid]['invalid_monotonic'] += stats['invalid_monotonic']
                global_segment_stats[sid]['invalid_bounds'] += stats['invalid_bounds']
                global_segment_stats[sid]['insufficient_records'] += stats['insufficient_records']
                
                global_segment_stats[sid]['sum_outlier_ratio'] += stats['sum_outlier_ratio']
                global_segment_stats[sid]['count_outlier_ratio'] += stats['count_outlier_ratio']
                global_segment_stats[sid]['sum_outlier_stop'] += stats['sum_outlier_stop']
                global_segment_stats[sid]['count_outlier_stop'] += stats['count_outlier_stop']
                
    total_sum_ratio = sum(s['sum_outlier_ratio'] for s in global_segment_stats.values())
    total_count_ratio = sum(s['count_outlier_ratio'] for s in global_segment_stats.values())
    total_sum_stop = sum(s['sum_outlier_stop'] for s in global_segment_stats.values())
    total_count_stop = sum(s['count_outlier_stop'] for s in global_segment_stats.values())
    
    file_avg_ratio = total_sum_ratio / total_count_ratio if total_count_ratio > 0 else 0.0
    file_avg_stop = total_sum_stop / total_count_stop if total_count_stop > 0 else 0.0

    top_ratio_segs = sorted(global_segment_stats.items(), key=lambda x: x[1]['invalid_outlier_ratio'], reverse=True)
    top_10_ratio = [(sid, stats['invalid_outlier_ratio']) for sid, stats in top_ratio_segs if stats['invalid_outlier_ratio'] > 0][:10]
    
    top_stop_segs = sorted(global_segment_stats.items(), key=lambda x: x[1]['invalid_outlier_stop'], reverse=True)
    top_10_stop = [(sid, stats['invalid_outlier_stop']) for sid, stats in top_stop_segs if stats['invalid_outlier_stop'] > 0][:10]
    
    logger.info("\n--- Top 10 Outlier Ratio Filled Segments ---")
    if top_10_ratio:
        for sid, count in top_10_ratio:
            logger.info(f"Segment {sid}: {count} traversals removed")
    else:
        logger.info("None")
        
    logger.info("\n--- Top 10 Outlier Stopping Segments ---")
    if top_10_stop:
        for sid, count in top_10_stop:
            logger.info(f"Segment {sid}: {count} traversals removed")
    else:
        logger.info("None")
        
    logger.info("\n--- File Outlier Statistics ---")
    logger.info(f"Average Outlier Proportion: {file_avg_ratio:.2%}")
    logger.info(f"Average Outlier Stop Duration: {file_avg_stop:.2f}s")
        
    top_prop_segs = sorted(
        global_segment_stats.items(), 
        key=lambda x: (x[1]['invalid'] / x[1]['total'] if x[1]['total'] > 0 else 0, x[1]['total']), 
        reverse=True
    )
    top_10_prop = [(sid, stats) for sid, stats in top_prop_segs if stats['invalid'] > 0][:10]
    
    logger.info("\n--- Top 10 Segments by Invalid Traversal Proportion ---")
    if top_10_prop:
        for sid, stats in top_10_prop:
            prop = stats['invalid'] / stats['total']
            logger.info(f"Segment {sid}: {prop:.1%} removed ({stats['invalid']}/{stats['total']})")
            logger.info(f"    - Outlier Ratio: {stats['invalid_outlier_ratio']}")
            logger.info(f"    - Outlier Stop: {stats['invalid_outlier_stop']}")
            logger.info(f"    - Gap Threshold: {stats['invalid_gap']}")
            logger.info(f"    - Monotonicity: {stats['invalid_monotonic']}")
            logger.info(f"    - Incomplete Traversal (Missing Bounds): {stats['invalid_bounds']}")
            logger.info(f"    - Insufficient Records (< 2): {stats['insufficient_records']}")
    else:
        logger.info("None")
    logger.info("\n")
            
    logger.info(f"Spline fitting and traversal extraction took {time.time() - t2:.2f}s")
    logger.info(f"Extracted {len(valid_traversals)} valid traversals.")
    
    if not valid_traversals:
        logger.info("No valid traversals found.")
        return 0.0, file_avg_ratio, file_avg_stop, list(global_segment_stats.keys()), list(global_segment_stats.keys())
        
    if config.get('update_traversals') and all_valid_indices:
        logger.info(f"Updating {len(all_valid_indices)} valid trajectory points to is_outlier=False...")
        df.loc[all_valid_indices, 'is_outlier'] = False
        updated_csv_path = out_dir / 'matched_trajectories_updated.csv'
        df.to_csv(updated_csv_path, index=False)
        logger.info(f"Saved updated trajectories to {updated_csv_path}")

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
                p5_time = np.percentile(times, config['percentile_temporal'])
                red_light_duration = np.percentile(seg_df['stopping_duration'], config['percentile_red_light'])
                if red_light_duration > file_max_red_light_duration:
                    file_max_red_light_duration = red_light_duration
                    file_max_red_light_segment_id = segment_id
                ideal_time = p5_time + red_light_duration
                if ideal_time > 0:
                    temp_thresh = seg_length / ideal_time
                else:
                    logger.info(f"Segment {segment_id} has non-positive ideal time for temporal threshold calculation (p5_time={p5_time}, red_light_duration={red_light_duration})")
            else:
                logger.info(f"Segment {segment_id} has no traversals longer than 2.0 seconds for temporal threshold calculation (count={len(times)}, length={seg_length})")
        else:
            logger.info(f"Segment {segment_id} has insufficient traversals for temporal threshold calculation (count={len(times)}, length={seg_length})")
                    
        # Spatial Threshold
        spat_thresh = None
        free_flow_speed = -1.0
        if len(seg_df) > 1 and temp_thresh is not None:
            good_temp_df = seg_df[seg_df['temporal_mean_speed'] >= temp_thresh]
            if len(good_temp_df) > 0:
                spat_thresh = np.percentile(good_temp_df['spatial_mean_speed'], config['percentile_spatial'])
                free_flow_speed = np.percentile(good_temp_df['spatial_mean_speed'], config['percentile_free_flow'])
            else:
                logger.info(f"Segment {segment_id} has no traversals meeting temporal threshold for spatial threshold calculation (count={len(good_temp_df)}, temp_thresh={temp_thresh}, length={seg_length})")
        else:
            logger.info(f"Segment {segment_id} has insufficient data for spatial threshold calculation (count={len(seg_df)}, temp_thresh={temp_thresh}, lenght={seg_length})")
                
        if temp_thresh is None or spat_thresh is None:
            logger.info(f"Segment {segment_id} has invalid thresholds. Total={global_segment_stats[segment_id]['total']} Invalid={global_segment_stats[segment_id]['invalid']}.  Adding with default thresholds.")
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
            logger.info(f"Segment {segment_id} has no valid traversals. Total={stats['total']}  Invalid={stats['invalid']}. Adding with default thresholds.")
            segment_thresholds[sid_int] = {
                'temporal_threshold': -1.0,
                'spatial_threshold': -1.0,
                'free_flow_speed': -1.0,
                'red_light_duration': 0.0,
                'total_traversals': stats['total'],
                'invalid_traversals': stats['invalid']
            }

    invalid_segments_in_file = []
    low_volume_segments_in_file = []
    for sid_int, thresholds in segment_thresholds.items():
        if thresholds['temporal_threshold'] == -1.0 or thresholds['spatial_threshold'] == -1.0:
            invalid_segments_in_file.append(str(sid_int))
            
        valid_count = thresholds['total_traversals'] - thresholds['invalid_traversals']
        if valid_count < 10:
            low_volume_segments_in_file.append(str(sid_int))

    logger.info(f"Threshold calculation took {time.time() - t3:.2f}s")
    
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
    logger.info(f"RTSM Calculation took {time.time() - t4:.2f}s")
    
    with open(out_dir / 'segment_thresholds.json', 'w') as f:
        json.dump(segment_thresholds, f, indent=4)
        
    out_cols = ['segment_id', 'track_id', 'timestamp', 'temporal_mean_speed', 'spatial_mean_speed', 'rtsm']
    traversals_df.sort_values('segment_id')[out_cols].to_csv(out_dir / 'traversal_metrics.csv', index=False)
        
    logger.info(f"Total time for {csv_path}: {time.time() - t0:.2f}s\n")
    
    if file_max_red_light_duration > 0:
        logger.info(f"Max red light duration for this file: {file_max_red_light_duration:.2f}s (Segment ID: {file_max_red_light_segment_id})\n")
        
    return file_max_red_light_duration, file_avg_ratio, file_avg_stop, invalid_segments_in_file, low_volume_segments_in_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract vehicle traversal metrics from matched trajectories.")
    parser.add_argument("--file", type=str, help="Path to a single matched_trajectories.csv file")
    parser.add_argument("--folder", type=str, help="Root directory to search recursively for matched_trajectories.csv")
    parser.add_argument("--update_traversals", action="store_true", help="Update valid traversals to have is_outlier=False and save to matched_trajectories_updated.csv")
    
    args = parser.parse_args()
    
    global_logger = logging.getLogger("global_traversal_metrics")
    global_logger.setLevel(logging.INFO)
    if global_logger.hasHandlers():
        global_logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    global_logger.addHandler(ch)
    
    if args.folder:
        global_log_path = Path(args.folder) / 'global-traversal-metrics.log'
        fh = logging.FileHandler(global_log_path, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(message)s'))
        global_logger.addHandler(fh)
    
    config = {
        "gap_threshold": 30.0,
        "min_edge_threshold": 8.0,
        "max_edge_threshold": 25.0,
        "edge_prop_threshold": 0.2,
        "connection_length_threshold": 5.0,
        "speed_sample_interval": 10.0,
        "stopped_speed_threshold": 0.1,
        "min_traversal_time": 2.0,
        "monotonicity_offset": 0.001,
        "percentile_temporal": 5,
        "percentile_red_light": 95,
        "percentile_spatial": 5,
        "percentile_free_flow": 85,
        "max_outlier_proportion": 0.50,
        "max_outlier_stop_duration": 5.0,
        "update_traversals": args.update_traversals
    }
    
    max_red_light_durations = []
    file_avg_ratios = []
    file_avg_stops = []
    invalid_segments_tracker = {}
    low_volume_segments_tracker = {}

    if args.file:
        csv_path = Path(args.file)
        if csv_path.exists():
            res = process_file(csv_path, config)
            if isinstance(res, tuple) and len(res) == 5:
                rl, a_ratio, a_stop, inv_segs, low_vol_segs = res
                if rl > 0:
                    max_red_light_durations.append(rl)
                file_avg_ratios.append(a_ratio)
                file_avg_stops.append(a_stop)
                for seg in inv_segs:
                    if seg not in invalid_segments_tracker:
                        invalid_segments_tracker[seg] = []
                    invalid_segments_tracker[seg].append(csv_path.name)
                for seg in low_vol_segs:
                    if seg not in low_volume_segments_tracker:
                        low_volume_segments_tracker[seg] = []
                    low_volume_segments_tracker[seg].append(csv_path.name)
    elif args.folder:
        root_path = Path(args.folder)
        # Search for all subdirectories containing either file
        for subdir in root_path.rglob("*"):
            if subdir.is_dir():
                filtered_path = subdir / "matched_trajectories_filtered.csv"
                unfiltered_path = subdir / "matched_trajectories.csv"
                
                target_path = None
                if filtered_path.exists():
                    target_path = filtered_path
                elif unfiltered_path.exists():
                    target_path = unfiltered_path
                
                if target_path:
                    res = process_file(target_path, config)
                    if isinstance(res, tuple) and len(res) == 5:
                        rl, a_ratio, a_stop, inv_segs, low_vol_segs = res
                        if rl > 0:
                            max_red_light_durations.append(rl)
                        file_avg_ratios.append(a_ratio)
                        file_avg_stops.append(a_stop)
                        for seg in inv_segs:
                            if seg not in invalid_segments_tracker:
                                invalid_segments_tracker[seg] = []
                            invalid_segments_tracker[seg].append(target_path.parent.name)
                        for seg in low_vol_segs:
                            if seg not in low_volume_segments_tracker:
                                low_volume_segments_tracker[seg] = []
                            low_volume_segments_tracker[seg].append(target_path.parent.name)
    else:
        global_logger.info("Please provide either --file or --folder")
        
    if len(max_red_light_durations) > 1:
        avg_max_red_light = sum(max_red_light_durations) / len(max_red_light_durations)
        global_logger.info(f"Average of max red light durations across {len(max_red_light_durations)} files: {avg_max_red_light:.2f}s")
        
    if len(file_avg_ratios) > 0:
        overall_avg_ratio = sum(file_avg_ratios) / len(file_avg_ratios)
        overall_avg_stop = sum(file_avg_stops) / len(file_avg_stops)
        global_logger.info(f"Average of outlier proportions across {len(file_avg_ratios)} files: {overall_avg_ratio:.2%}")
        global_logger.info(f"Average of outlier stop durations across {len(file_avg_stops)} files: {overall_avg_stop:.2f}s")
        
    if invalid_segments_tracker:
        global_logger.info("\n--- Invalid Segments Across Files ---")
        for seg, files in sorted(invalid_segments_tracker.items(), key=lambda x: len(x[1]), reverse=True):
            global_logger.info(f"Segment {seg} was invalid in {len(files)} files:")
            for f in files:
                global_logger.info(f"  - {f}")
                
    if low_volume_segments_tracker:
        global_logger.info("\n--- Low Volume Segments (< 10 Valid Traversals) ---")
        for seg, files in sorted(low_volume_segments_tracker.items(), key=lambda x: len(x[1]), reverse=True):
            global_logger.info(f"Segment {seg} had < 10 valid traversals in {len(files)} files:")
            for f in files:
                global_logger.info(f"  - {f}")