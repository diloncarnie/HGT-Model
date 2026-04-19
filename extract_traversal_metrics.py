import pandas as pd
import numpy as np
import geopandas as gpd
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

CHAIN_CRITICAL_TAG = " [CHAIN - CRITICAL]"
ROAD_TYPE_CRITICAL_TAG = " [PRIMARY/SECONDARY - CRITICAL]"
PRIMARY_SECONDARY_TYPES = {'primary', 'secondary', 'primary_link', 'secondary_link'}

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

def _distance_outlier_ratio(tproj, outlier_mask, seg_length):
    """Fraction of segment length 'covered' by outlier points via Voronoi
    half-intervals in t_proj space. Independent of sample density."""
    if seg_length <= 0:
        return 0.0
    n = len(tproj)
    if n == 0:
        return 0.0
    order = np.argsort(tproj, kind='stable')
    tp = np.clip(tproj[order].astype(float), 0.0, float(seg_length))
    om = outlier_mask[order]
    if n == 1:
        return float(om[0])
    bounds = np.empty(n + 1, dtype=float)
    bounds[0] = 0.0
    bounds[-1] = float(seg_length)
    bounds[1:-1] = (tp[:-1] + tp[1:]) / 2.0
    widths = np.clip(np.diff(bounds), 0.0, None)
    return float(widths[om].sum() / seg_length)


def _max_run_duration(mask_both, dt):
    """Max sum of dt[i] over runs where mask_both[i] stays True."""
    if not mask_both.any():
        return 0.0
    current = 0.0
    mx = 0.0
    for i in range(len(mask_both)):
        if mask_both[i]:
            current += dt[i]
            if current > mx:
                mx = current
        else:
            current = 0.0
    return mx


def _compute_chain_segment_ids(net_gdf):
    """
    Returns the set of segment_ids (as strings) that are interior chain segments:
    both endpoints are shared with at least one other segment in the network.
    A dead-end segment has at least one endpoint that no other segment touches.
    """
    from collections import defaultdict
    node_to_segs = defaultdict(set)

    for _, row in net_gdf.iterrows():
        seg_id = str(row['segment_id'])
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            coords = list(geom.coords)
        except Exception:
            continue
        if len(coords) < 2:
            continue
        start = (round(coords[0][0], 7), round(coords[0][1], 7))
        end   = (round(coords[-1][0], 7), round(coords[-1][1], 7))
        node_to_segs[start].add(seg_id)
        node_to_segs[end].add(seg_id)

    chain_ids = set()
    for _, row in net_gdf.iterrows():
        seg_id = str(row['segment_id'])
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            coords = list(geom.coords)
        except Exception:
            continue
        if len(coords) < 2:
            continue
        start = (round(coords[0][0], 7), round(coords[0][1], 7))
        end   = (round(coords[-1][0], 7), round(coords[-1][1], 7))
        if len(node_to_segs[start]) > 1 and len(node_to_segs[end]) > 1:
            chain_ids.add(seg_id)

    return chain_ids


def _compute_primary_secondary_ids(net_gdf):
    """
    Returns the set of segment_ids (as strings) whose highway tag is
    primary, secondary, primary_link, or secondary_link.
    """
    if 'highway' not in net_gdf.columns or 'segment_id' not in net_gdf.columns:
        return set()
    mask = net_gdf['highway'].astype(str).str.lower().isin(PRIMARY_SECONDARY_TYPES)
    return set(net_gdf.loc[mask, 'segment_id'].astype(str))


def _new_segment_stat():
    return {
        'total': 0, 'invalid': 0, 'invalid_outlier_ratio': 0,
        'invalid_outlier_stop': 0, 'invalid_gap': 0,
        'invalid_monotonic': 0, 'invalid_bounds': 0, 'insufficient_records': 0,
        'sum_outlier_ratio': 0.0, 'count_outlier_ratio': 0,
        'sum_outlier_stop': 0.0, 'count_outlier_stop': 0
    }


def process_track(track_data, config):
    """Processes all segments for a single track_id using numpy arrays."""
    track_id, track_df = track_data

    track_df = track_df.copy()
    track_df['original_index'] = track_df.index
    track_df = track_df.sort_values('time').reset_index(drop=True)

    traversals = []
    segment_stats = {}
    logs = []
    valid_original_indices = []

    n = len(track_df)
    if n == 0:
        return traversals, segment_stats, logs, valid_original_indices

    seg_ids = track_df['segment_id'].values
    times_all = track_df['time'].values.astype(float)
    speeds_all = track_df['speed'].values.astype(float)
    tproj_all = track_df['t_proj'].values.astype(float)
    seglen_all = track_df['segment_length'].values.astype(float)
    orig_idx_all = track_df['original_index'].values
    has_outlier_col = 'is_outlier' in track_df.columns
    outlier_all = track_df['is_outlier'].values.astype(bool) if has_outlier_col else None

    # Group boundaries on consecutive runs of same segment_id
    change = np.empty(n, dtype=bool)
    change[0] = True
    if n > 1:
        change[1:] = seg_ids[1:] != seg_ids[:-1]
    boundaries = np.where(change)[0]
    ends = np.append(boundaries[1:], n)

    gap_thresh = config['gap_threshold']
    stopped_thresh = config['stopped_speed_threshold']
    max_outlier_prop = config.get('max_outlier_proportion', 0.2)
    max_outlier_stop_dur = config.get('max_outlier_stop_duration', 5.0)
    min_edge = config['min_edge_threshold']
    max_edge = config['max_edge_threshold']
    edge_prop = config['edge_prop_threshold']
    mono_off = config['monotonicity_offset']
    speed_int = config['speed_sample_interval']

    for start, end in zip(boundaries, ends):
        if end <= start:
            continue
        segment_id = seg_ids[start]
        if segment_id not in segment_stats:
            segment_stats[segment_id] = _new_segment_stat()
        st = segment_stats[segment_id]
        st['total'] += 1

        segment_length = float(seglen_all[start])
        g_times = times_all[start:end]
        g_speeds = speeds_all[start:end]
        g_tproj = tproj_all[start:end]
        g_orig = orig_idx_all[start:end]
        g_len = end - start

        max_outlier_stop = 0.0
        if has_outlier_col:
            g_outlier = outlier_all[start:end]
            outlier_ratio = _distance_outlier_ratio(g_tproj, g_outlier, segment_length)
            st['sum_outlier_ratio'] += outlier_ratio
            st['count_outlier_ratio'] += 1

            stopped_mask = g_outlier & (g_speeds < stopped_thresh)
            if stopped_mask.any() and g_len > 1:
                both = stopped_mask[1:] & stopped_mask[:-1]
                max_outlier_stop = _max_run_duration(both, np.diff(g_times))
                st['sum_outlier_stop'] += float(max_outlier_stop)
                st['count_outlier_stop'] += 1

            if outlier_ratio > max_outlier_prop:
                st['invalid'] += 1
                st['invalid_outlier_ratio'] += 1
                continue
            if max_outlier_stop > max_outlier_stop_dur:
                st['invalid'] += 1
                st['invalid_outlier_stop'] += 1
                continue

        if g_len > 1 and (np.diff(g_times) > gap_thresh).any():
            st['invalid'] += 1
            st['invalid_gap'] += 1
            continue

        cummax = np.maximum.accumulate(g_tproj)
        valid_mask = g_tproj >= cummax
        if not valid_mask.any():
            logs.append(f"Track {track_id}, segment {segment_id} has no valid monotonic records. Skipping.")
            st['invalid'] += 1
            st['invalid_monotonic'] += 1
            continue

        valid_rel = np.where(valid_mask)[0]
        first_rel, last_rel = valid_rel[0], valid_rel[-1]
        first_abs, last_abs = start + first_rel, start + last_rel
        vg_times = g_times[valid_mask]
        vg_speeds = g_speeds[valid_mask]
        vg_tproj = g_tproj[valid_mask]

        edge_t = max(min_edge, min(max_edge, edge_prop * segment_length))

        prev_tp = prev_time = prev_speed = None
        next_tp = next_time = next_speed = None
        has_prev = False
        has_following = False

        if first_abs > 0:
            if seg_ids[first_abs - 1] != segment_id:
                prev_tp = -(float(seglen_all[first_abs - 1]) - float(tproj_all[first_abs - 1]))
                prev_time = float(times_all[first_abs - 1])
                prev_speed = float(speeds_all[first_abs - 1])
                has_prev = True
        elif vg_tproj[0] <= edge_t:
            has_prev = True

        if last_abs < n - 1:
            if seg_ids[last_abs + 1] != segment_id:
                next_tp = segment_length + float(tproj_all[last_abs + 1])
                next_time = float(times_all[last_abs + 1])
                next_speed = float(speeds_all[last_abs + 1])
                has_following = True
        elif segment_length - vg_tproj[-1] <= edge_t:
            has_following = True

        n_records = len(vg_tproj) + (1 if prev_tp is not None else 0) + (1 if next_tp is not None else 0)
        if not has_prev or not has_following or n_records < 2:
            st['invalid'] += 1
            if n_records < 2:
                st['insufficient_records'] += 1
            else:
                st['invalid_bounds'] += 1
            continue

        parts_tp = []
        parts_time = []
        parts_speed = []
        if prev_tp is not None:
            parts_tp.append(np.array([prev_tp]))
            parts_time.append(np.array([prev_time]))
            parts_speed.append(np.array([prev_speed]))
        parts_tp.append(vg_tproj)
        parts_time.append(vg_times)
        parts_speed.append(vg_speeds)
        if next_tp is not None:
            parts_tp.append(np.array([next_tp]))
            parts_time.append(np.array([next_time]))
            parts_speed.append(np.array([next_speed]))

        t_proj_arr = np.concatenate(parts_tp).astype(float)
        time_arr = np.concatenate(parts_time).astype(float)
        speed_arr = np.concatenate(parts_speed).astype(float)

        # Enforce strict monotonicity by offset — same as original
        for i in range(1, len(t_proj_arr)):
            min_allowed = t_proj_arr[i - 1] + mono_off
            if t_proj_arr[i] < min_allowed:
                t_proj_arr[i] = min_allowed

        try:
            length = segment_length
            t_at_0 = float(np.interp(0.0, t_proj_arr, time_arr))
            t_at_L = float(np.interp(length, t_proj_arr, time_arr))
            traversal_time = t_at_L - t_at_0

            if traversal_time <= 0:
                temporal_mean_speed = float(vg_speeds.mean())
            else:
                temporal_mean_speed = length / traversal_time

            first_offset = max(0.0, float(t_proj_arr[0]))
            last_offset = min(length, float(t_proj_arr[-1]))
            cur = math.ceil(first_offset)
            end_sp = math.floor(last_offset)

            if end_sp - cur < speed_int:
                spatial_mean_speed = float(vg_speeds.mean())
            else:
                n_pts = int((end_sp - cur) // speed_int) + 1
                eval_points = cur + speed_int * np.arange(n_pts, dtype=float)
                eval_points = eval_points[(end_sp - eval_points) >= speed_int]
                if len(eval_points) == 0:
                    spatial_mean_speed = float(vg_speeds.mean())
                else:
                    sp_vals = np.interp(eval_points, t_proj_arr, speed_arr)
                    spatial_mean_speed = float(sp_vals.mean())
        except Exception as e:
            logs.append(f"Interpolation error for track {track_id}, segment {segment_id}: {e}")
            total_dist = float(t_proj_arr[-1] - t_proj_arr[0])
            total_t = float(time_arr[-1] - time_arr[0])
            if total_t > 0:
                temporal_mean_speed = total_dist / total_t
            else:
                temporal_mean_speed = float(vg_speeds.mean())
            spatial_mean_speed = float(vg_speeds.mean())
            traversal_time = total_t

        if len(vg_speeds) > 1:
            stopped = vg_speeds < stopped_thresh
            both_s = stopped[1:] & stopped[:-1]
            stopping_duration = _max_run_duration(both_s, np.diff(vg_times))
        else:
            stopping_duration = 0.0

        record = {
            'segment_id': segment_id,
            'track_id': track_id,
            'timestamp': vg_times[-1],
            'temporal_mean_speed': temporal_mean_speed,
            'spatial_mean_speed': spatial_mean_speed,
            'stopping_duration': stopping_duration,
            'traversal_time': traversal_time,
            'segment_length': segment_length
        }
        traversals.append(record)
        valid_original_indices.extend(g_orig.tolist())

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
        return 0.0, 0.0, 0.0, [], [], [], [], [], [], [], []

    logger.info(f"Loaded {len(df)} rows. Time: {time.time() - t0:.2f}s")

    # Detect parked vehicles (stationary for >= 120s)
    df['is_parked'] = False
    stopped_thresh = config.get('stopped_speed_threshold', 0.1)
    parked_thresh = 120.0
    all_parked_indices = []

    logger.info("Detecting parked vehicles...")
    t_parked = time.time()
    
    for track_id, track_df in df.groupby('track_id'):
        track_df = track_df.sort_values('time')
        is_stopped = track_df['speed'].values < stopped_thresh
        if not is_stopped.any():
            continue
            
        times = track_df['time'].values
        change_points = np.diff(is_stopped.astype(int))
        run_starts = np.where(change_points == 1)[0] + 1
        if is_stopped[0]:
            run_starts = np.insert(run_starts, 0, 0)
            
        run_ends = np.where(change_points == -1)[0]
        if is_stopped[-1]:
            run_ends = np.append(run_ends, len(is_stopped) - 1)
            
        for start_idx, end_idx in zip(run_starts, run_ends):
            duration = times[end_idx] - times[start_idx]
            if duration >= parked_thresh:
                all_parked_indices.extend(track_df.index[start_idx:end_idx+1].tolist())

    if all_parked_indices:
        df.loc[all_parked_indices, 'is_parked'] = True
        logger.info(f"Flagged {len(all_parked_indices)} trajectory points as parked. Time: {time.time() - t_parked:.2f}s")
    else:
        logger.info(f"No parked vehicles detected. Time: {time.time() - t_parked:.2f}s")

    # Controller protection diagnostic: report how many points sit near a
    # controller vs how many are flagged as outliers, as a sanity check.
    if 'dist_to_controller' in df.columns:
        near_ctrl = (df['dist_to_controller'] <= config['signal_proximity_threshold']).sum()
        total_outliers = int(df['is_outlier'].sum()) if 'is_outlier' in df.columns else 0
        logger.info(
            f"Controller proximity: {near_ctrl} trajectory points within 5 m "
            f"of a controller signal; {total_outliers} points flagged as outliers."
        )

    t1 = time.time()

    # Load empirical free flow speeds to merge into thresholds
    empirical_speeds_path = out_dir / 'empirical_free_flow_speeds.json'
    empirical_speeds = {}
    if empirical_speeds_path.exists():
        with open(empirical_speeds_path, 'r') as f:
            empirical_speeds = json.load(f)
    else:
        logger.warning(f"Could not find {empirical_speeds_path}. Falling back to default values where necessary.")

    # Load internal junction segment IDs from the network gpkg in the same directory.
    # Prefer the filtered network if it exists, otherwise fall back to the base one.
    junction_segment_ids = set()
    chain_segment_ids = set()
    primary_secondary_ids = set()
    for gpkg_name in ('osm_network_filtered.gpkg', 'osm_network.gpkg'):
        gpkg_path = out_dir / gpkg_name
        if gpkg_path.exists():
            try:
                net = gpd.read_file(gpkg_path)
                if 'is_internal_junction' in net.columns and 'segment_id' in net.columns:
                    mask = net['is_internal_junction'].astype(str).str.lower() == 'true'
                    junction_segment_ids = set(net.loc[mask, 'segment_id'].astype(str))
                    logger.info(f"Loaded {len(junction_segment_ids)} internal junction segment IDs from {gpkg_name}.")
                chain_segment_ids = _compute_chain_segment_ids(net)
                logger.info(f"Identified {len(chain_segment_ids)} interior chain segments in {gpkg_name}.")
                primary_secondary_ids = _compute_primary_secondary_ids(net)
                logger.info(f"Identified {len(primary_secondary_ids)} primary/secondary segments in {gpkg_name}.")
            except Exception as e:
                logger.warning(f"Could not read {gpkg_path} for junction IDs: {e}")
            break

    if junction_segment_ids:
        ij_mask = df['segment_id'].astype(str).isin(junction_segment_ids)
        ij_count = int(ij_mask.sum())
        if ij_count > 0:
            logger.info(f"Skipping {ij_count} trajectory points on internal junction segments.")
            work_df = df.loc[~ij_mask].copy()
        else:
            work_df = df
    else:
        work_df = df

    # Process by track_id to allow matching previous/following records correctly
    grouped = work_df.groupby('track_id')
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
            chain_tag = CHAIN_CRITICAL_TAG if str(sid) in chain_segment_ids else ""
            rt_tag = ROAD_TYPE_CRITICAL_TAG if str(sid) in primary_secondary_ids else ""
            logger.info(f"Segment {sid}{chain_tag}{rt_tag}: {prop:.1%} removed ({stats['invalid']}/{stats['total']})")
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
    
    if config.get('update_traversals'):
        if all_valid_indices:
            logger.info(f"Updating {len(all_valid_indices)} valid trajectory points to is_outlier=False...")
            df.loc[all_valid_indices, 'is_outlier'] = False
        updated_csv_path = out_dir / 'matched_trajectories_updated.csv'
        df.to_csv(updated_csv_path, index=False)
        logger.info(f"Saved updated trajectories to {updated_csv_path}")

    if not valid_traversals:
        logger.info("No valid traversals found.")
        return (0.0, file_avg_ratio, file_avg_stop,
                list(global_segment_stats.keys()), list(global_segment_stats.keys()),
                [], [], [], [], [], [])
        
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
            
        segment_thresholds[str(segment_id)] = {
            'temporal_threshold': temp_thresh,
            'spatial_threshold': spat_thresh,
            'free_flow_speed': free_flow_speed,
            'empirical_free_flow_speed': empirical_speeds.get(str(segment_id), -1.0),
            'red_light_duration': red_light_duration,
            'total_traversals': global_segment_stats[segment_id]['total'],
            'invalid_traversals': global_segment_stats[segment_id]['invalid']
        }

    # Add segments that only had invalid traversals
    for segment_id, stats in global_segment_stats.items():
        
        sid_str = str(segment_id)
        if sid_str not in segment_thresholds:
            logger.info(f"Segment {segment_id} has no valid traversals. Total={stats['total']}  Invalid={stats['invalid']}. Adding with default thresholds.")
            segment_thresholds[sid_str] = {
                'temporal_threshold': -1.0,
                'spatial_threshold': -1.0,
                'free_flow_speed': -1.0,
                'empirical_free_flow_speed': empirical_speeds.get(str(segment_id), -1.0),
                'red_light_duration': 0.0,
                'total_traversals': stats['total'],
                'invalid_traversals': stats['invalid']
            }

    invalid_segments_in_file = []
    low_volume_segments_in_file = []
    high_volume_segments_in_file = []
    chain_invalid_in_file = []
    chain_low_volume_in_file = []
    ps_invalid_in_file = []
    ps_low_volume_in_file = []
    for sid_str, thresholds in segment_thresholds.items():
        if thresholds['temporal_threshold'] == -1.0 or thresholds['spatial_threshold'] == -1.0:
            invalid_segments_in_file.append(sid_str)
            if sid_str in chain_segment_ids:
                chain_invalid_in_file.append(sid_str)
            if sid_str in primary_secondary_ids:
                ps_invalid_in_file.append(sid_str)

        valid_count = thresholds['total_traversals'] - thresholds['invalid_traversals']
        if valid_count < 10:
            low_volume_segments_in_file.append(sid_str)
            if sid_str in chain_segment_ids:
                chain_low_volume_in_file.append(sid_str)
            if sid_str in primary_secondary_ids:
                ps_low_volume_in_file.append(sid_str)
        else:
            high_volume_segments_in_file.append(sid_str)

    logger.info("\n--- Chain Segments with Issues (Interior Connectors) ---")
    if chain_invalid_in_file:
        logger.info(f"[CHAIN - CRITICAL] {len(chain_invalid_in_file)} interior chain segment(s) have invalid thresholds:")
        for sid in sorted(chain_invalid_in_file):
            stats = segment_thresholds[str(sid)]
            rt_tag = ROAD_TYPE_CRITICAL_TAG if sid in primary_secondary_ids else ""
            logger.info(f"  Segment {sid}{rt_tag}: total={stats['total_traversals']} invalid={stats['invalid_traversals']}")
    else:
        logger.info("No invalid interior chain segments.")
    if chain_low_volume_in_file:
        logger.info(f"[CHAIN - CRITICAL] {len(chain_low_volume_in_file)} interior chain segment(s) have < 10 valid traversals:")
        for sid in sorted(chain_low_volume_in_file):
            stats = segment_thresholds[str(sid)]
            valid_count = stats['total_traversals'] - stats['invalid_traversals']
            rt_tag = ROAD_TYPE_CRITICAL_TAG if sid in primary_secondary_ids else ""
            logger.info(f"  Segment {sid}{rt_tag}: valid={valid_count} total={stats['total_traversals']}")
    else:
        logger.info("No low-volume interior chain segments.")

    logger.info("\n--- Primary/Secondary Segments with Issues ---")
    if ps_invalid_in_file:
        logger.info(f"[PRIMARY/SECONDARY - CRITICAL] {len(ps_invalid_in_file)} primary/secondary segment(s) have invalid thresholds:")
        for sid in sorted(ps_invalid_in_file):
            stats = segment_thresholds[str(sid)]
            chain_tag = CHAIN_CRITICAL_TAG if sid in chain_segment_ids else ""
            logger.info(f"  Segment {sid}{chain_tag}: total={stats['total_traversals']} invalid={stats['invalid_traversals']}")
    else:
        logger.info("No invalid primary/secondary segments.")
    if ps_low_volume_in_file:
        logger.info(f"[PRIMARY/SECONDARY - CRITICAL] {len(ps_low_volume_in_file)} primary/secondary segment(s) have < 10 valid traversals:")
        for sid in sorted(ps_low_volume_in_file):
            stats = segment_thresholds[str(sid)]
            valid_count = stats['total_traversals'] - stats['invalid_traversals']
            chain_tag = CHAIN_CRITICAL_TAG if sid in chain_segment_ids else ""
            logger.info(f"  Segment {sid}{chain_tag}: valid={valid_count} total={stats['total_traversals']}")
    else:
        logger.info("No low-volume primary/secondary segments.")

    # Per-file summary by segment category
    general_invalid_count = len([s for s in invalid_segments_in_file if s not in chain_segment_ids and s not in primary_secondary_ids])
    general_low_volume_count = len([s for s in low_volume_segments_in_file if s not in chain_segment_ids and s not in primary_secondary_ids])

    logger.info(f"\n--- File Segment Summary ---")
    logger.info(f"  Chain links:                  {len(chain_invalid_in_file)} invalid, {len(chain_low_volume_in_file)} low-volume (< 10 valid)")
    logger.info(f"  Primary/Secondary:            {len(ps_invalid_in_file)} invalid, {len(ps_low_volume_in_file)} low-volume (< 10 valid)")
    logger.info(f"  General (non-chain, non-P/S): {general_invalid_count} invalid, {general_low_volume_count} low-volume (< 10 valid)")

    logger.info(f"Threshold calculation took {time.time() - t3:.2f}s")
    
    t4 = time.time()
    # RTSM Calculation (vectorized)
    sid_arr = traversals_df['segment_id'].astype(str).values
    temp_arr = traversals_df['temporal_mean_speed'].values.astype(float)
    spat_arr = traversals_df['spatial_mean_speed'].values.astype(float)
    t_thresh_arr = np.array([segment_thresholds[s]['temporal_threshold'] for s in sid_arr], dtype=float)
    s_thresh_arr = np.array([segment_thresholds[s]['spatial_threshold'] for s in sid_arr], dtype=float)

    rtsm_values = np.zeros(len(sid_arr), dtype=float)
    invalid_mask = (t_thresh_arr == -1.0) | (s_thresh_arr == -1.0)
    worst_case = t_thresh_arr + s_thresh_arr
    zero_wc = (worst_case == 0) & ~invalid_mask
    unknown_speed = ((temp_arr == -1) | (spat_arr == -1)) & ~invalid_mask & ~zero_wc
    active = ~invalid_mask & ~zero_wc & ~unknown_speed

    if active.any():
        t = temp_arr[active]
        s = spat_arr[active]
        tt = t_thresh_arr[active]
        ss = s_thresh_arr[active]
        t_ok = t >= tt
        s_ok = s >= ss
        dist = np.where(
            s_ok,
            np.where(t_ok, 0.0, tt - t),
            np.where(t_ok, ss - s, (tt - t) + (ss - s))
        )
        rtsm_active = dist / worst_case[active]
        rtsm_values[active] = np.clip(rtsm_active, 0.0, 1.0)

    rtsm_values[invalid_mask] = -1.0
    rtsm_values[zero_wc] = 0.0
    rtsm_values[unknown_speed] = 1.0

    traversals_df['rtsm'] = rtsm_values

    # Identify segments whose RTSM defaulted to -1
    no_rtsm_segment_ids = set(traversals_df.loc[traversals_df['rtsm'] == -1.0, 'segment_id'].astype(str))
    no_rtsm_count = int((rtsm_values == -1.0).sum())
    logger.info(f"\n--- Segments with No RTSM (defaulted to -1) ---")
    logger.info(f"{len(no_rtsm_segment_ids)} segment(s) could not calculate RTSM ({no_rtsm_count} traversals affected).")
    if no_rtsm_segment_ids:
        for sid in sorted(no_rtsm_segment_ids):
            stats = segment_thresholds.get(str(sid), {})
            chain_tag = CHAIN_CRITICAL_TAG if sid in chain_segment_ids else ""
            rt_tag = ROAD_TYPE_CRITICAL_TAG if sid in primary_secondary_ids else ""
            logger.info(f"  Segment {sid}{chain_tag}{rt_tag}: total={stats.get('total_traversals', 0)} invalid={stats.get('invalid_traversals', 0)}")

    logger.info(f"RTSM Calculation took {time.time() - t4:.2f}s")
    
    with open(out_dir / 'segment_thresholds.json', 'w') as f:
        json.dump(segment_thresholds, f, indent=4)
        
    out_cols = ['segment_id', 'track_id', 'timestamp', 'temporal_mean_speed', 'spatial_mean_speed', 'rtsm']
    traversals_df.sort_values('segment_id')[out_cols].to_csv(out_dir / 'traversal_metrics.csv', index=False)
        
    logger.info(f"Total time for {csv_path}: {time.time() - t0:.2f}s\n")
    
    if file_max_red_light_duration > 0:
        logger.info(f"Max red light duration for this file: {file_max_red_light_duration:.2f}s (Segment ID: {file_max_red_light_segment_id})\n")
        
    return (file_max_red_light_duration, file_avg_ratio, file_avg_stop,
            invalid_segments_in_file, low_volume_segments_in_file,
            chain_invalid_in_file, chain_low_volume_in_file,
            ps_invalid_in_file, ps_low_volume_in_file,
            list(no_rtsm_segment_ids), high_volume_segments_in_file)

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
        "gap_threshold": 15.0,
        "min_edge_threshold": 10.0,
        "max_edge_threshold": 35.0,
        "edge_prop_threshold": 0.5,
        "connection_length_threshold": 5.0,
        "speed_sample_interval": 10.0,
        "stopped_speed_threshold": 0.1,
        "min_traversal_time": 2.0,
        "monotonicity_offset": 0.001,
        "percentile_temporal": 5,
        "percentile_red_light": 95,
        "percentile_spatial": 5,
        "percentile_free_flow": 85,
        "max_outlier_proportion": 0.80,
        "max_outlier_stop_duration": 25.0,
        "signal_proximity_threshold":5.0,
        "update_traversals": args.update_traversals
    }
    
    max_red_light_durations = []
    file_avg_ratios = []
    file_avg_stops = []
    invalid_segments_tracker = {}
    low_volume_segments_tracker = {}
    chain_invalid_tracker = {}
    chain_low_volume_tracker = {}
    ps_invalid_tracker = {}
    ps_low_volume_tracker = {}
    no_rtsm_tracker = {}
    high_volume_segments = set()

    def _accumulate_results(res, label):
        if not (isinstance(res, tuple) and len(res) == 11):
            return
        rl, a_ratio, a_stop, inv_segs, low_vol_segs, chain_inv, chain_low, ps_inv, ps_low, no_rtsm, high_vol = res
        if rl > 0:
            max_red_light_durations.append(rl)
        file_avg_ratios.append(a_ratio)
        file_avg_stops.append(a_stop)
        for seg in inv_segs:
            invalid_segments_tracker.setdefault(seg, []).append(label)
        for seg in low_vol_segs:
            low_volume_segments_tracker.setdefault(seg, []).append(label)
        for seg in chain_inv:
            chain_invalid_tracker.setdefault(seg, []).append(label)
        for seg in chain_low:
            chain_low_volume_tracker.setdefault(seg, []).append(label)
        for seg in ps_inv:
            ps_invalid_tracker.setdefault(seg, []).append(label)
        for seg in ps_low:
            ps_low_volume_tracker.setdefault(seg, []).append(label)
        for seg in no_rtsm:
            no_rtsm_tracker.setdefault(seg, []).append(label)
        high_volume_segments.update(high_vol)

        # Per-file segment summary in global log
        general_inv_count = len([s for s in inv_segs if s not in set(chain_inv) and s not in set(ps_inv)])
        general_low_count = len([s for s in low_vol_segs if s not in set(chain_low) and s not in set(ps_low)])
        global_logger.info(f"\n--- File Segment Summary: {label} ---")
        global_logger.info(f"  Chain links:                  {len(chain_inv)} invalid, {len(chain_low)} low-volume (< 10 valid)")
        global_logger.info(f"  Primary/Secondary:            {len(ps_inv)} invalid, {len(ps_low)} low-volume (< 10 valid)")
        global_logger.info(f"  General (non-chain, non-P/S): {general_inv_count} invalid, {general_low_count} low-volume (< 10 valid)")

    if args.file:
        csv_path = Path(args.file)
        if csv_path.exists():
            _accumulate_results(process_file(csv_path, config), csv_path.name)
    elif args.folder:
        root_path = Path(args.folder)
        for subdir in root_path.rglob("*"):
            if subdir.is_dir():
                filtered_path = subdir / "matched_trajectories_filtered.csv"
                unfiltered_path = subdir / "matched_trajectories.csv"
                if filtered_path.exists():
                    target_path = filtered_path
                elif unfiltered_path.exists():
                    target_path = unfiltered_path
                else:
                    target_path = None
                if target_path:
                    _accumulate_results(process_file(target_path, config), target_path.parent.name)
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

    # Classify each segment into exactly one category:
    # Priority: CHAIN > PRIMARY/SECONDARY > general
    # Within each category: invalid > low-volume (no overlap)
    all_chain_segs = set(chain_invalid_tracker) | set(chain_low_volume_tracker)
    # Primary/secondary excludes any segment already in chain
    ps_only_invalid = {s: f for s, f in ps_invalid_tracker.items() if s not in all_chain_segs}
    ps_only_low_volume = {s: f for s, f in ps_low_volume_tracker.items() if s not in all_chain_segs}
    all_categorized = all_chain_segs | set(ps_only_invalid) | set(ps_only_low_volume)
    # Low-volume excludes invalid within each category
    chain_low_only = {s: f for s, f in chain_low_volume_tracker.items() if s not in chain_invalid_tracker}
    ps_low_only = {s: f for s, f in ps_only_low_volume.items() if s not in ps_only_invalid}
    # General sections exclude segments already in chain or primary/secondary
    general_invalid = {s: f for s, f in invalid_segments_tracker.items() if s not in all_categorized}
    general_low_volume = {s: f for s, f in low_volume_segments_tracker.items()
                          if s not in all_categorized and s not in general_invalid}

    if chain_invalid_tracker or chain_low_only:
        global_logger.info("\n=== CHAIN SEGMENTS WITH ISSUES (Interior Connectors) ===")
        if chain_invalid_tracker:
            global_logger.info("\n--- [CHAIN - CRITICAL] Invalid Interior Chain Segments Across Files ---")
            for seg, files in sorted(chain_invalid_tracker.items(), key=lambda x: len(x[1]), reverse=True):
                global_logger.info(f"Segment {seg}{CHAIN_CRITICAL_TAG} was invalid in {len(files)} file(s):")
                for f in files:
                    global_logger.info(f"  - {f}")
        if chain_low_only:
            global_logger.info("\n--- [CHAIN - CRITICAL] Low-Volume Interior Chain Segments Across Files ---")
            for seg, files in sorted(chain_low_only.items(), key=lambda x: len(x[1]), reverse=True):
                global_logger.info(f"Segment {seg}{CHAIN_CRITICAL_TAG} had < 10 valid traversals in {len(files)} file(s):")
                for f in files:
                    global_logger.info(f"  - {f}")

    if ps_only_invalid or ps_low_only:
        global_logger.info("\n=== PRIMARY/SECONDARY SEGMENTS WITH ISSUES ===")
        if ps_only_invalid:
            global_logger.info("\n--- [PRIMARY/SECONDARY - CRITICAL] Invalid Primary/Secondary Segments Across Files ---")
            for seg, files in sorted(ps_only_invalid.items(), key=lambda x: len(x[1]), reverse=True):
                global_logger.info(f"Segment {seg}{ROAD_TYPE_CRITICAL_TAG} was invalid in {len(files)} file(s):")
                for f in files:
                    global_logger.info(f"  - {f}")
        if ps_low_only:
            global_logger.info("\n--- [PRIMARY/SECONDARY - CRITICAL] Low-Volume Primary/Secondary Segments Across Files ---")
            for seg, files in sorted(ps_low_only.items(), key=lambda x: len(x[1]), reverse=True):
                global_logger.info(f"Segment {seg}{ROAD_TYPE_CRITICAL_TAG} had < 10 valid traversals in {len(files)} file(s):")
                for f in files:
                    global_logger.info(f"  - {f}")

    if general_invalid:
        global_logger.info("\n--- Invalid Segments Across Files [Non-Primary/Secondary and Non-Chain Links] ---")
        for seg, files in sorted(general_invalid.items(), key=lambda x: len(x[1]), reverse=True):
            global_logger.info(f"Segment {seg} was invalid in {len(files)} files:")
            for f in files:
                global_logger.info(f"  - {f}")

    if general_low_volume:
        global_logger.info("\n--- Low Volume Segments (< 10 Valid Traversals) ---")
        for seg, files in sorted(general_low_volume.items(), key=lambda x: len(x[1]), reverse=True):
            global_logger.info(f"Segment {seg} had < 10 valid traversals in {len(files)} files:")
            for f in files:
                global_logger.info(f"  - {f}")

    if no_rtsm_tracker:
        global_logger.info(f"\n=== Segments with No RTSM (defaulted to -1) ===")
        global_logger.info(f"{len(no_rtsm_tracker)} segment(s) could not calculate RTSM across all files.")
        for seg, files in sorted(no_rtsm_tracker.items(), key=lambda x: len(x[1]), reverse=True):
            chain_tag = CHAIN_CRITICAL_TAG if seg in chain_invalid_tracker or seg in chain_low_volume_tracker else ""
            rt_tag = ROAD_TYPE_CRITICAL_TAG if seg in ps_invalid_tracker or seg in ps_low_volume_tracker else ""
            global_logger.info(f"Segment {seg}{chain_tag}{rt_tag} had no RTSM in {len(files)} file(s):")
            for f in files:
                global_logger.info(f"  - {f}")

    # --- Build removed_segment_traversals and update global networks ---
    if args.folder:
        num_files = len(file_avg_ratios)
        half_files = num_files / 2.0
        removed_segment_traversals = set()

        # Non-Primary/Secondary and Non-Chain segments invalid in more than half the files
        for seg, files in general_invalid.items():
            if len(files) > half_files:
                removed_segment_traversals.add(seg)
                global_logger.info(f"Removing general segment {seg}: invalid in {len(files)}/{num_files} files")

        # Non-Primary/Secondary and Non-Chain segments with < 10 valid traversals in more than half the files
        for seg, files in low_volume_segments_tracker.items():
            if seg in all_categorized:
                continue
            if len(files) > half_files:
                removed_segment_traversals.add(seg)
                global_logger.info(f"Removing general segment {seg}: < 10 valid traversals in {len(files)}/{num_files} files")

        # PRIMARY/SECONDARY segments (non-chain) invalid in more than half the files
        for seg, files in ps_only_invalid.items():
            if len(files) > half_files:
                removed_segment_traversals.add(seg)
                global_logger.info(f"Removing primary/secondary segment {seg}: invalid in {len(files)}/{num_files} files")

        # PRIMARY/SECONDARY segments (non-chain) with < 10 valid traversals in ALL files
        # A segment qualifies if it was never high-volume (>= 10 valid) in any file
        for seg, files in ps_only_low_volume.items():
            if seg not in high_volume_segments:
                removed_segment_traversals.add(seg)
                global_logger.info(f"Removing primary/secondary segment {seg}: < 10 valid traversals in all files (low-volume in {len(files)}/{num_files} files)")

        root_path = Path(args.folder)
        removed_traversals_path = root_path / 'removed_segment_traversals.json'
        with open(removed_traversals_path, 'w') as f:
            json.dump(sorted(removed_segment_traversals), f, indent=4)
        global_logger.info(f"\nWrote {len(removed_segment_traversals)} segments to {removed_traversals_path}")

        if removed_segment_traversals:
            for gpkg_name in ('osm_network.gpkg', 'osm_network_common.gpkg'):
                gpkg_path = root_path / gpkg_name
                if gpkg_path.exists():
                    try:
                        gdf = gpd.read_file(gpkg_path)
                        if 'segment_id' in gdf.columns:
                            filtered = gdf[~gdf['segment_id'].astype(str).isin(removed_segment_traversals)]
                            updated_name = gpkg_name.replace('.gpkg', '_updated.gpkg')
                            updated_path = root_path / updated_name
                            filtered.to_file(updated_path, driver='GPKG')
                            global_logger.info(f"Saved {updated_path} ({len(gdf)} -> {len(filtered)} segments)")
                        else:
                            global_logger.warning(f"{gpkg_path} has no segment_id column, skipping.")
                    except Exception as e:
                        global_logger.error(f"Error filtering {gpkg_path}: {e}")
        else:
            global_logger.info("No segments to remove based on traversal data.")