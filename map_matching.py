import argparse
import time
import os
import shutil
import json
import multiprocessing
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jenkspy
from shapely.geometry import Point, LineString
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap

def get_osm_map(edges_gdf):
    map_con = InMemMap("osm", use_latlon=False, use_rtree=True, index_edges=True)
    for row in edges_gdf.itertuples():
        u, v = int(row.u), int(row.v)
        coords = list(row.geometry.coords)
        map_con.add_node(u, (coords[0][0], coords[0][1]))
        map_con.add_node(v, (coords[-1][0], coords[-1][1]))
        map_con.add_edge(u, v)
    return map_con

def _parse_pneuma_chunk(args):
    lines_chunk, config = args
    data_list = []
    variance = int(config["sampling_interval"] * 0.1)  # 10% variance
    for line in lines_chunk:
        parts = line.strip().split(';')
        if len(parts) < 4: continue
        
        track_id = parts[0].strip()
        v_type = parts[1].strip()
        
        if v_type == 'Motorcycle':
            continue
            
        traveled_d = parts[2].strip()
        avg_speed_kmh = float(parts[3].strip())
        avg_speed_ms = avg_speed_kmh / 3.6
        
        points = parts[4:]
        sampled_pts = []
        last_included_timestamp = None
        
        for i in range(0, len(points)-1, 6):
            current_gap = config["sampling_interval"] + np.random.randint(-variance, variance + 1)
            try:
                lat, lon, speed_kmh, lon_acc, lat_acc, ts = points[i:i+6]
                ts_float = float(ts.strip())
                timestamp_ms = int(ts_float * 1000)

                if last_included_timestamp is None or timestamp_ms - last_included_timestamp >= current_gap:
                    speed_ms = float(speed_kmh.strip()) / 3.6
                    sampled_pts.append([
                        float(lat.strip()), float(lon.strip()), speed_ms,
                        float(lon_acc.strip()), float(lat_acc.strip()), ts_float
                    ])
                    last_included_timestamp = timestamp_ms
            except ValueError:
                continue
        
        # Filter duration < 5.0s, traveled_distance < 20m, and avg_speed < 1m/s
        if (len(sampled_pts) > 0 and 
            sampled_pts[-1][-1] - sampled_pts[0][-1] >= 5.0 and
            float(traveled_d) >= 20.0 and
            avg_speed_ms >= 1.0):
            for p in sampled_pts:
                data_list.append([track_id, v_type, traveled_d, avg_speed_ms] + p)
                
    columns = ['track_id', 'type', 'traveled_d', 'avg_speed', 'lat', 'lon', 'speed', 'lon_acc', 'lat_acc', 'time']
    if not data_list:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(data_list, columns=columns)

def parse_pneuma_to_long(filepath, config, logger, test=False):
    logger.info(f"Parsing {filepath}...")
    start_time = time.time()
    
    if test:
        # Quick count total non-motorcycle vehicles to take percentage
        with open(filepath, 'r', encoding='utf-8') as f_count:
            first = f_count.readline()
            has_header = "track_id" in first
            num_vehicles = 0
            if not has_header:
                f_count.seek(0)
            
            for line in f_count:
                line = line.strip()
                if not line: continue
                parts = line.split(';')
                if len(parts) > 1 and parts[1].strip() != 'Motorcycle':
                    num_vehicles += 1
                    
        limit = max(1, int(num_vehicles * config["test_percentage"]))
        logger.info(f"Test mode: taking {config['test_percentage']*100:.1f}% of non-motorcycle vehicles ({limit} out of {num_vehicles})")
    else:
        limit = float('inf')

    def chunk_generator():
        with open(filepath, 'r') as f:
            first_line = f.readline()
            if "track_id" not in first_line:
                f.seek(0)
                
            chunk = []
            count = 0
            for line in f:
                if count >= limit:
                    break
                chunk.append(line)
                count += 1
                
                # 500 trajectories per chunk is a good balance for IPC overhead vs worker load
                if len(chunk) >= 500:
                    yield (chunk, config)
                    chunk = []
                    
            if chunk:
                yield (chunk, config)

    df_list = []
    num_processes = multiprocessing.cpu_count()
    
    # Use imap_unordered to process chunks as they are generated and yielded
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result_df in pool.imap_unordered(_parse_pneuma_chunk, chunk_generator()):
            if not result_df.empty:
                df_list.append(result_df)

    if df_list:
        df = pd.concat(df_list, ignore_index=True)
    else:
        columns = ['track_id', 'type', 'traveled_d', 'avg_speed', 'lat', 'lon', 'speed', 'lon_acc', 'lat_acc', 'time']
        df = pd.DataFrame(columns=columns)
    
    logger.info(f"Parsing took {time.time() - start_time:.2f} seconds. Parsed {len(df)} points.")
    return df

# Global variables for worker processes to prevent repeated memory allocation
global_map_con = None
global_edge_data = None
global_lanes_map = None
global_len_map = None
global_uv_to_seg = None
global_hw_map = None
global_seg_to_controller = None
global_controller_signals = None
global_config = None

def _map_match_worker_init(gpkg_path, config):
    global global_map_con, global_edge_data, global_lanes_map, global_len_map
    global global_uv_to_seg, global_hw_map, global_seg_to_controller
    global global_controller_signals, global_config
    global_config = config

    edges_gdf = gpd.read_file(gpkg_path)
    global_map_con = get_osm_map(edges_gdf)

    global_edge_data = {}
    global_lanes_map = {}
    global_len_map = {}
    global_uv_to_seg = {}
    global_hw_map = {}

    # Load controller data for controller-aware outlier protection
    global_seg_to_controller = {}
    global_controller_signals = {}
    controllers_path = os.path.join(os.path.dirname(gpkg_path) or '.', 'controllers.json')
    if os.path.exists(controllers_path):
        with open(controllers_path, 'r') as f:
            controllers = json.load(f)
        for ctrl_id, ctrl in controllers.items():
            sig_pts = np.array(
                [[s['x_utm'], s['y_utm']] for s in ctrl.get('raw_osm_signals', [])],
                dtype=float
            ).reshape(-1, 2) if ctrl.get('raw_osm_signals') else np.empty((0, 2), dtype=float)
            global_controller_signals[ctrl_id] = sig_pts
            for sid in ctrl.get('approach_segments', []):
                global_seg_to_controller[sid] = ctrl_id
            for sid in ctrl.get('junction_segments', []):
                if sid not in global_seg_to_controller:
                    global_seg_to_controller[sid] = ctrl_id

    for row in edges_gdf.itertuples():
        key = (int(row.u), int(row.v))
        try:
            if pd.isna(row.lanes):
                global_lanes_map[key] = 2
            else:
                global_lanes_map[key] = int(row.lanes)
        except Exception:
            global_lanes_map[key] = 2
        global_len_map[key] = float(row.length)
        global_uv_to_seg[key] = str(row.segment_id)
        global_hw_map[key] = str(row.highway)

        coords = np.array(row.geometry.coords)
        if len(coords) < 2:
            continue
            
        a = coords[:-1]
        b = coords[1:]
        ab = b - a
        sub_l2 = np.sum(ab * ab, axis=1)
        sub_l = np.sqrt(sub_l2)
        accum_l = np.insert(np.cumsum(sub_l), 0, 0)[:-1]
        valid = sub_l2 > 0
        az_sub = np.arctan2(ab[:, 0], ab[:, 1]) * 180 / np.pi
        
        global_edge_data[key] = {
            'a': a, 'ab': ab, 'sub_l2': sub_l2, 'sub_l': sub_l,
            'accum_l': accum_l, 'valid': valid, 'az_sub': az_sub
        }

def _map_match_track(track_tuple):
    track_id, group = track_tuple
    path = group[['x', 'y']].values.tolist()
    
    matcher = None
    matched_states = None
    for max_dist in range(global_config["map_matching_max_dist_start"], global_config["map_matching_max_dist_end"] + 1, global_config["map_matching_step"]):
        matcher = DistanceMatcher(global_map_con, max_dist=max_dist, obs_noise=max_dist, non_emitting_states=False)
        states, last_idx = matcher.match(path)
        if states and (last_idx == len(path) - 1 or max_dist == global_config["map_matching_max_dist_end"]):
            matched_states = states
            break
            
    if not matched_states:
        return pd.DataFrame()
        
    group_results = group.iloc[:len(matched_states)].copy()
    
    u_list = []
    v_list = []
    for s in matched_states:
        if isinstance(s, tuple) and len(s) >= 2:
            u_list.append(int(s[0]))
            v_list.append(int(s[1]))
        else:
            u_list.append(int(s))
            v_list.append(int(s))
            
    group_results['matched_u'] = u_list
    group_results['matched_v'] = v_list
    
    # Calculate vehicle azimuth 
    dx = group_results['x'].diff().bfill().fillna(0).values
    dy = group_results['y'].diff().bfill().fillna(0).values
    group_results['azcar'] = np.arctan2(dx, dy) * 180 / np.pi
    
    # Optimizing the point projection loop by grouping by edge and mapping in 3D matrices
    edge_groups = {}
    x_vals = group_results['x'].values
    y_vals = group_results['y'].values
    azcar_vals = group_results['azcar'].values
    
    for idx in range(len(group_results)):
        key = (u_list[idx], v_list[idx])
        if key not in edge_groups:
            edge_groups[key] = []
        edge_groups[key].append(idx)
        
    signed_distances = np.zeros(len(group_results))
    segment_ids = np.empty(len(group_results), dtype=object)
    num_lanes_list = np.zeros(len(group_results), dtype=int)
    segment_lengths = np.zeros(len(group_results))
    prop_distances = np.full(len(group_results), 0.5)
    rel_headings = np.zeros(len(group_results))
    highways_list = np.empty(len(group_results), dtype=object)
    t_projs = np.zeros(len(group_results))
    controller_id_list = np.empty(len(group_results), dtype=object)
    dist_to_controller_list = np.full(len(group_results), np.inf)

    for key, indices in edge_groups.items():
        if key not in global_edge_data:
            for idx in indices:
                segment_ids[idx] = ""
                highways_list[idx] = ""
                controller_id_list[idx] = ""
            continue

        ed = global_edge_data[key]
        a, ab = ed['a'], ed['ab']
        sub_l2, sub_l, accum_l = ed['sub_l2'], ed['sub_l'], ed['accum_l']
        valid, az_sub = ed['valid'], ed['az_sub']

        total_l = global_len_map.get(key, accum_l[-1] + sub_l[-1] if len(accum_l) > 0 else 1.0)
        if total_l <= 0.0: total_l = 1.0

        seg_id = global_uv_to_seg.get(key, str(key))
        n_lanes = global_lanes_map.get(key, 2)
        hw = global_hw_map.get(key, "unknown")

        # Controller-aware signal proximity
        ctrl_id = global_seg_to_controller.get(seg_id, "")
        ctrl_signals = global_controller_signals.get(ctrl_id) if ctrl_id else None

        # Vectorized Projections
        p = np.column_stack((x_vals[indices], y_vals[indices]))
        ap = p[:, np.newaxis, :] - a[np.newaxis, :, :]
        dot_ap_ab = ap[:, :, 0] * ab[:, 0] + ap[:, :, 1] * ab[:, 1]

        t = np.zeros_like(dot_ap_ab)
        t[:, valid] = np.clip(dot_ap_ab[:, valid] / sub_l2[valid], 0, 1)

        proj = a[np.newaxis, :, :] + t[:, :, np.newaxis] * ab[np.newaxis, :, :]
        diff = p[:, np.newaxis, :] - proj
        dists = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)

        best_idx = np.argmin(dists, axis=1)
        row_idx = np.arange(len(indices))
        best_dist = dists[row_idx, best_idx]

        cross = ab[best_idx, 0] * ap[row_idx, best_idx, 1] - ab[best_idx, 1] * ap[row_idx, best_idx, 0]
        signed_d = best_dist * np.where(cross > 0, -1, 1)

        dist_along = accum_l[best_idx] + t[row_idx, best_idx] * sub_l[best_idx]
        rel_h = np.abs((azcar_vals[indices] - az_sub[best_idx] + 180) % 360 - 180)

        # Euclidean distance from each point to the nearest raw signal of
        # the controller this segment belongs to.
        if ctrl_signals is not None and ctrl_signals.shape[0] > 0:
            diffs_ctrl = p[:, np.newaxis, :] - ctrl_signals[np.newaxis, :, :]
            ctrl_dists = np.sqrt(diffs_ctrl[:, :, 0]**2 + diffs_ctrl[:, :, 1]**2).min(axis=1)
        else:
            ctrl_dists = np.full(len(indices), np.inf)

        for i, idx in enumerate(indices):
            signed_distances[idx] = signed_d[i]
            segment_ids[idx] = seg_id
            num_lanes_list[idx] = n_lanes
            segment_lengths[idx] = total_l
            prop_distances[idx] = dist_along[i] / total_l
            highways_list[idx] = hw
            t_projs[idx] = dist_along[i]
            rel_headings[idx] = rel_h[i]
            controller_id_list[idx] = ctrl_id
            dist_to_controller_list[idx] = ctrl_dists[i]

    group_results['signed_dist'] = signed_distances
    group_results['segment_id'] = segment_ids
    group_results['num_lanes'] = num_lanes_list
    group_results['segment_length'] = segment_lengths
    group_results['prop_dist'] = prop_distances
    group_results['rel_heading'] = rel_headings
    group_results['highway'] = highways_list
    group_results['t_proj'] = t_projs
    group_results['controller_id'] = controller_id_list
    group_results['dist_to_controller'] = dist_to_controller_list
    
    group_results = group_results[group_results['rel_heading'] <= global_config["rel_heading_limit"]].copy()
    
    if group_results.empty:
        return pd.DataFrame()
        
    def is_fully_traversed(grp):
        if grp['segment_length'].iloc[0] < global_config["partial_traversal_length_thresh"]:
            return True
        return (grp['prop_dist'].max() - grp['prop_dist'].min()) >= global_config["partial_traversal_prop_thresh"]

    valid_traversals = group_results.groupby('segment_id').filter(is_fully_traversed)
    return valid_traversals

def perform_map_matching(df, crs, gpkg_path, config, logger):
    logger.info("Performing map matching and distance calculation with pool load balancing...")
    start_time = time.time()
    
    # Project to CRS
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    gdf = gdf.to_crs(crs)
    df['x'] = gdf.geometry.x
    df['y'] = gdf.geometry.y
    
    grouped = df.groupby('track_id')
    num_processes = multiprocessing.cpu_count()
    
    results = []
    with multiprocessing.Pool(processes=num_processes, initializer=_map_match_worker_init, initargs=(gpkg_path, config)) as pool:
        for res in pool.imap_unordered(_map_match_track, grouped):
            if not res.empty:
                results.append(res)
    
    final_df = pd.concat([r for r in results if not r.empty], ignore_index=True) if results else pd.DataFrame()
    logger.info(f"Map matching and distance calculation took {time.time() - start_time:.2f} seconds. Matched {len(final_df)} points.")
    return final_df

def filter_hq_d_values(d_vals, speeds, rel_headings, config):
    """Filter D values to a high-quality subset using progressive speed thresholds."""
    for s_thresh in config["jenks_speed_thresholds"]:
        mask = (speeds > s_thresh) & (rel_headings < config["jenks_heading_threshold"])
        if mask.sum() >= config["jenks_min_points"]:
            return d_vals[mask]
    return d_vals


def density_prioritized_sample(hq_d, config):
    """Generate a sample that favors more dense bins for the Jenks algorithm."""
    if len(hq_d) == 0:
        return hq_d

    sampling_budget = min(len(hq_d), config.get("jenks_max_target_calc", 15000))
    if len(hq_d) <= sampling_budget:
        return hq_d

    counts, bins = np.histogram(hq_d, bins=config.get("jenks_bins", 40))
    
    # Exaggerate density using a power > 1. 
    # density_power = 1.0 is strictly proportional. 
    # density_power = 2.0 squares the counts, heavily biasing the sample towards the densest bins.
    density_power = config.get("jenks_density_power", 1.0)
    weights = np.power(counts, density_power)
    
    # Avoid division by zero if all weights become 0
    if np.sum(weights) == 0:
        bin_probs = np.ones(len(counts)) / len(counts)
    else:
        bin_probs = weights / np.sum(weights)

    # Allocate the sampling budget across bins based on the skewed probabilities
    samples_per_bin = np.round(bin_probs * sampling_budget).astype(int)

    samples = []

    for i in range(len(counts)):
        if counts[i] == 0:
            continue
        bin_mask = (hq_d >= bins[i]) & (hq_d <= bins[i+1])
        bin_vals = hq_d[bin_mask]
        
        # Take the allocated amount, capping at the actual number of points in the bin
        num_to_take = min(samples_per_bin[i], len(bin_vals))
        
        if num_to_take > 0:
            samples.extend(np.random.choice(bin_vals, num_to_take, replace=False))
            
    return np.array(samples)


def jenks_breaks_with_fallback(sample, k, road_min, road_max):
    """Compute Jenks breaks for a known k. Falls back to equal-width breaks on failure."""
    if k <= 1:
        return [road_min, road_max]
    try:
        return list(jenkspy.jenks_breaks(sample, n_classes=k))
    except ValueError:
        max_distance = max(road_max - road_min, 1e-6)
        return [road_min + i * (max_distance / k) for i in range(k + 1)]


def calculate_signed_distance_and_lanes(df, edges_gdf, output_dir, config, logger, test=False):
    if test:
        logger.info("Performing Jenks Optimization...")
    start_time = time.time()

    # Calculate and print max samples
    if test:
        counts = df[df['segment_id'] != ""]['segment_id'].value_counts()
        if not counts.empty:
            logger.info(f"Maximum samples in a single segment: {counts.max()}")

    # Process Jenks
    lane_boundaries = {}
    jenks_segments_used = []
    failed_segments = []

    grouped = df.groupby('segment_id')

    for seg_id, group in grouped:
        if seg_id == "": continue

        d_vals = group['signed_dist'].values
        
        # Determine percentiles for robust road width estimation (default 2 and 98)
        lower_p, upper_p = 2, 98
        if seg_id in config.get("segment_boundaries", {}):
            custom_p = config["segment_boundaries"][seg_id]
            if isinstance(custom_p, list) and len(custom_p) == 2:
                lower_p, upper_p = custom_p[0], custom_p[1]
                
        abs_max_d = np.percentile(d_vals, upper_p)
        abs_min_d = np.percentile(d_vals, lower_p)
        road_width = abs_max_d - abs_min_d

        # D = Right_Edge - Vehicle_Distance. Clip extreme outliers on both sides.
        df.loc[group.index, 'D'] = (abs_max_d - group['signed_dist']).clip(0.0, road_width)
        D_vals = df.loc[group.index, 'D'].values

        num_lanes_osm = int(group['num_lanes'].iloc[0])

        if len(D_vals) > config["jenks_min_points"]:
            hq_D_final = filter_hq_d_values(D_vals, group['speed'].values, group['rel_heading'].values, config)
            if test:
                logger.info(f"Segment {seg_id}: Using {len(hq_D_final)} HQ points for Jenks.")

            road_min, road_max = hq_D_final.min(), hq_D_final.max()
            sample = density_prioritized_sample(hq_D_final, config)

            # Iterative Lane Partitioning: Start with max_distance / avg_lane_width
            max_distance = (road_max - road_min)  
            avg_lane_width = config["avg_lane_width"]
            
            is_link_highway = str(group['highway'].iloc[0]).endswith('_link')
            seg_edge = edges_gdf[edges_gdf['segment_id'].astype(str) == str(seg_id)]
            is_internal_junc = False
            has_osm_lanes = False
            
            if not seg_edge.empty:
                if 'is_internal_junction' in seg_edge.columns:
                    is_internal_junc = str(seg_edge['is_internal_junction'].iloc[0]).lower() == 'true'
                if 'lanes' in seg_edge.columns:
                    has_osm_lanes = pd.notna(seg_edge['lanes'].iloc[0])
                
            is_link = is_link_highway or is_internal_junc
            
            if is_link and has_osm_lanes:
                initial_k = max(1, num_lanes_osm)
            else:
                initial_k = max(1, round(max_distance / avg_lane_width))

            best_breaks = None
            if seg_id in config["segment_lanes"]:
                k = config["segment_lanes"][seg_id]
                if k == 1:
                    best_breaks = [road_min, road_max]
                    if test:
                        logger.info(f"Segment {seg_id}: Manual override applied. Assigned 1 lane [road_min, road_max].")
                else:
                    try:
                        best_breaks = jenkspy.jenks_breaks(sample, n_classes=k)
                        if test:
                            logger.info(f"Segment {seg_id}: Manual override applied. Assigned {k} lanes using Jenks.")
                    except ValueError:
                        best_breaks = [road_min + i * (max_distance / k) for i in range(k + 1)]
                        failed_segments.append(seg_id)
                        if test:
                            logger.info(f"Segment {seg_id}: Manual override applied. Jenks failed, using equal widths for {k} lanes.")
            elif is_link:
                if initial_k == 1:
                    best_breaks = [road_min, road_max]
                    if test:
                        logger.info(f"Segment {seg_id}: Link segment detected. Bypassed validation, assigned 1 lane [road_min, road_max].")
                else:
                    try:
                        best_breaks = jenkspy.jenks_breaks(sample, n_classes=initial_k)
                        if test:
                            logger.info(f"Segment {seg_id}: Link segment detected. Bypassed validation, assigned {initial_k} lanes using Jenks.")
                    except ValueError:
                        best_breaks = [road_min + i * (max_distance / initial_k) for i in range(initial_k + 1)]
                        failed_segments.append(seg_id)
                        if test:
                            logger.info(f"Segment {seg_id}: Link segment detected. Jenks failed, using equal widths for {initial_k} lanes.")
            elif initial_k == 1:
                # Informed decision for 1 lane: Try Jenks for 2 or 3 lanes.
                for k in [2, 3]:
                    try:
                        breaks = jenkspy.jenks_breaks(sample, n_classes=k)
                        valid = True
                        for i in range(1, len(breaks)):
                            width = breaks[i] - breaks[i-1]
                            if width < config["min_lane_width_loose"] or width > config["max_lane_width"]:
                                valid = False
                                break
                        if valid:
                            best_breaks = breaks
                            if test:
                                logger.info(f"Segment {seg_id}: Increased 1-lane guess to {k} lanes based on Jenks.")
                            break
                    except ValueError:
                        continue
                
                # If neither 2 nor 3 lanes worked, default to road_min and road_max
                if best_breaks is None:
                    best_breaks = [road_min, road_max]
                    if test:
                        logger.info(f"Segment {seg_id}: Defaulted to 1 lane [road_min, road_max].")
            else:
                k = initial_k
                max_k = initial_k + 2  # Allow up to 2 more lanes than initial guess if needed
                
                while k <= max_k:
                    try:
                        breaks = jenkspy.jenks_breaks(sample, n_classes=k)
                        # Validate widths: lanes must be between 2.5m and 4.0m wide
                        valid = True
                        for i in range(1, len(breaks)):
                            width = breaks[i] - breaks[i-1]
                            if width < config["min_lane_width_strict"] or width > config["max_lane_width"]:
                                valid = False
                                break
                        if valid:
                            best_breaks = breaks
                            if test:
                                logger.info(f"Segment {seg_id}: Determined {k} lanes (Initial guess: {initial_k}).")
                            break
                        else:
                            k += 1 # Keep partitioning into more lanes
                    except ValueError:
                        break
                
                # If still no valid configuration, try reducing the number of lanes (up to 2 lower)
                if best_breaks is None:
                    for k_red in [initial_k - 1, initial_k - 2]:
                        if k_red < 1: continue
                        try:
                            breaks = jenkspy.jenks_breaks(sample, n_classes=k_red)
                            valid = True
                            for i in range(1, len(breaks)):
                                width = breaks[i] - breaks[i-1]
                                if width < config["min_lane_width_strict"] or width > config["max_lane_width"]:
                                    valid = False
                                    break
                            if valid:
                                best_breaks = breaks
                                if test:
                                    logger.info(f"Segment {seg_id}: Reduced to {k_red} lanes (Initial guess: {initial_k}).")
                                break
                        except ValueError:
                            continue

            if best_breaks:
                lane_boundaries[seg_id] = best_breaks
                jenks_segments_used.append(seg_id)
            else:
                # Fallback to initial guess
                try:
                    lane_boundaries[seg_id] = jenkspy.jenks_breaks(sample, n_classes=initial_k)
                    jenks_segments_used.append(seg_id)
                    failed_segments.append(seg_id)
                    if test:
                        logger.info(f"Segment {seg_id}: Jenks failed to find valid breaks. Used initial guess with {initial_k} lanes.")
                except ValueError:
                    lane_boundaries[seg_id] = [road_min + i * avg_lane_width for i in range(initial_k + 1)]
        else:
            if seg_id in config["segment_lanes"]:
                num_lanes_fixed = config["segment_lanes"][seg_id]
                lane_boundaries[seg_id] = [i * config["avg_lane_width"] for i in range(num_lanes_fixed + 1)]
                if test:
                    logger.info(f"Segment {seg_id}: Manual override applied. Assigned {num_lanes_fixed} lanes (low sample fallback).")
            else:
                lane_boundaries[seg_id] = [i * config["avg_lane_width"] for i in range(num_lanes_osm + 1)]

    # Lane Assignment and Outlier Removal
    df['lane_index'] = 0
    df['is_outlier'] = False
    
    lanes_updated_count = 0
    detected_lanes = {}
    for seg_id, bounds in lane_boundaries.items():
        mask = df['segment_id'] == seg_id
        if not mask.any(): continue
            
        raw_indices = np.digitize(df.loc[mask, 'D'], bins=bounds)

        # Identify outliers (indices 0 for < bounds[0] or len(bounds) for >= bounds[-1])
        seg_outliers = (raw_indices == 0) | (raw_indices == len(bounds))

        # Protect points near a traffic-signal controller. dist_to_controller is
        # the Euclidean distance from this point to the nearest raw signal of the
        # controller the matched segment belongs to; inf if no controller.
        if 'dist_to_controller' in df.columns:
            seg_ctrl_dist = df.loc[mask, 'dist_to_controller'].values
            near_controller = seg_ctrl_dist <= config["signal_proximity_threshold"]
            seg_outliers = seg_outliers & (~near_controller)

        df.loc[df[mask].index[seg_outliers], 'is_outlier'] = True
        
        # Map valid points to 0-based lane indices (clipping here just avoids OutOfBounds errors before removal)
        df.loc[mask, 'lane_index'] = np.clip(raw_indices, 1, len(bounds) - 1) - 1
        
        # Update num_lanes to match the number of detected Jenks lanes (or the original OSM lanes if Jenks wasn't used)
        old_lanes = df.loc[mask, 'num_lanes'].iloc[0]
        new_lanes = len(bounds) - 1
        detected_lanes[seg_id] = new_lanes
        if old_lanes != new_lanes:
            lanes_updated_count += 1
            
        df.loc[mask, 'num_lanes'] = new_lanes

    if lanes_updated_count > 0:
        logger.info(f"Successfully updated 'num_lanes' on {lanes_updated_count} segments using empirical Jenks data.")
    else:
        logger.info("No segments had their 'num_lanes' altered from the OSM default (likely due to low sample size or Jenks agreeing with OSM).")

    flagged_outliers = df['is_outlier'].sum()
    logger.info(f"Flagged {flagged_outliers} outlier trajectory points that did not fit into edge lanes.")

    logger.info(f"Jenks Optimization took {time.time() - start_time:.2f} seconds.")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'lane_boundaries.json'), 'w') as f:
        json.dump(lane_boundaries, f, indent=4)        
    
    # Plot debug histograms for specific segments
    debug_dir = os.path.join(output_dir, 'debugged_segments')
    failed_dir = os.path.join(debug_dir, 'failed_segments')
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    segments_to_plot = set(config["debug_segments"]).union(set(failed_segments))
    logger.info(f"Number of failed segments to plot: {len(failed_segments)}.")
    
    for seg in segments_to_plot:
        if seg in lane_boundaries:
            plt.figure(figsize=(8,4))
            seg_group = df[df['segment_id'] == seg]
            d_data = seg_group['D']
            n_lanes = len(lane_boundaries[seg]) - 1
            plt.hist(d_data, bins=50, alpha=0.7)
            for b in lane_boundaries[seg]:
                plt.axvline(b, color='r', linestyle='dashed', linewidth=2)
            plt.title(f'Lane Boundaries for Segment {seg} (Detected Lanes: {n_lanes})')
            plt.xlabel('Distance from Right Edge (m)')
            plt.ylabel('Frequency')
            
            if seg in failed_segments:
                save_path = os.path.join(failed_dir, f'histogram_seg_{seg}.png')
            else:
                save_path = os.path.join(debug_dir, f'histogram_seg_{seg}.png')
                
            plt.savefig(save_path)
            plt.close()
        
    return df, detected_lanes

def calculate_empirical_speeds(df, edges_gdf, output_dir, config, logger):
    logger.info("Calculating empirical speeds...")
    start_time = time.time()
    
    speed_df = df
    
    # Calculate traversal speeds (simply 95th percentile of observed speeds per segment)
    speeds = speed_df.groupby('segment_id')['speed'].quantile(0.95).to_dict()
    
    # Impute missing
    # Default highway speeds in m/s
    default_speeds = config["default_speeds"]
    
    empirical_speeds = {}
    for row in edges_gdf.itertuples():
        seg_id = str(row.segment_id)
        hw = row.highway
        if seg_id in speeds:
            empirical_speeds[seg_id] = speeds[seg_id]
        else:
            empirical_speeds[seg_id] = default_speeds.get(hw, config["default_speed_fallback"])
            
    with open(os.path.join(output_dir, 'empirical_free_flow_speeds.json'), 'w') as f:
        json.dump(empirical_speeds, f, indent=4)
        
    logger.info(f"Speed calculation took {time.time() - start_time:.2f} seconds.")
    return empirical_speeds

def process_single_file(input_file, edges_gdf, network_path, args, config, global_logger):
    file_start_time = time.time()
    
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    date_part = file_name.split('_')[0]
    output_dir = os.path.join("processed_data", date_part, file_name)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'map-matching.log')
    logger = logging.getLogger(f"logger_{file_name}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info(f"\n--- Processing {input_file} ---")
    global_logger.info(f"Started processing {input_file}")
    
    df = parse_pneuma_to_long(input_file, config, logger, test=args.test)
    if df.empty:
        logger.info("No data parsed.")
        global_logger.info(f"Skipped {input_file}: No data parsed.")
        return {}
        
    df = perform_map_matching(df, edges_gdf.crs, network_path, config, logger)
    if df.empty:
        global_logger.info(f"Skipped {input_file}: No matched trajectories.")
        return {}
        
    # Filter segments with fewer than 10 unique vehicles or in removed_segments
    logger.info(f"Filtering segments (Min {config['min_vehicles_per_segment']} vehicles, Exclude blacklist: {len(config['removed_segments'])})...")
    seg_counts = df[df['segment_id'] != ""].groupby('segment_id')['track_id'].nunique()
    
    # Identify segments to remove due to blacklist
    blacklisted_present = [s for s in seg_counts.index if s in config["removed_segments"]]
    points_in_blacklisted = len(df[df['segment_id'].isin(blacklisted_present)])
    
    # Identify segments to remove due to low volume (that are not already blacklisted)
    low_volume_segments = [s for s in seg_counts.index if s not in config["removed_segments"] and seg_counts[s] < config["min_vehicles_per_segment"]]
    points_in_low_volume = len(df[df['segment_id'].isin(low_volume_segments)])
    
    logger.info(f"Removed {points_in_blacklisted} points across {len(blacklisted_present)} blacklisted segments.")
    logger.info(f"Removed {points_in_low_volume} points across {len(low_volume_segments)} low volume segments (< {config['min_vehicles_per_segment']} vehicles).")
    
    # Identify segments that meet the volume threshold AND are not in the blacklist
    valid_segments = [s for s in seg_counts.index if s not in config["removed_segments"] and seg_counts[s] >= config["min_vehicles_per_segment"]]
    
    valid_segments_set = set(valid_segments)
    
    # Rule 1: Include shape junction links if they connect to a shape where at least one segment is valid
    if 'is_shape_junction' in edges_gdf.columns:
        shape_edges = edges_gdf[edges_gdf['is_shape_junction'].astype(str).str.lower() == 'true']
        if not shape_edges.empty:
            G_shape = nx.MultiGraph()
            for _, row in shape_edges.iterrows():
                G_shape.add_edge(row['u'], row['v'], segment_id=str(row['segment_id']))
                
            for component in nx.connected_components(G_shape):
                subgraph = G_shape.subgraph(component)
                comp_segment_ids = {data['segment_id'] for _, _, _, data in subgraph.edges(keys=True, data=True)}
                
                # If any segment in this shape component is in the currently valid segments, keep the whole shape
                if comp_segment_ids.intersection(valid_segments_set):
                    valid_segments_set.update(comp_segment_ids)

    # Rule 2: Remove fully disconnected valid segments (isolated edges or isolated two-way pairs)
    valid_edges = edges_gdf[edges_gdf['segment_id'].astype(str).isin(valid_segments_set)]
    G_valid = nx.MultiGraph()
    for _, row in valid_edges.iterrows():
        G_valid.add_edge(row['u'], row['v'], segment_id=str(row['segment_id']))
        
    isolated_segments = set()
    for component in nx.connected_components(G_valid):
        if len(component) <= 2:  # Component only has 1 or 2 nodes (isolated segment)
            subgraph = G_valid.subgraph(component)
            for u, v, key, data in subgraph.edges(keys=True, data=True):
                isolated_segments.add(data['segment_id'])
                
    if isolated_segments:
        valid_segments_set -= isolated_segments
        logger.info(f"Removed {len(isolated_segments)} fully disconnected valid segments.")
        
    # Rule 3: Remove segments with no topological influence
    adjacency = config.get("topological_adjacency", {})
    if adjacency:
        while True:
            isolated_topo_segments = set()
            valid_str_set = {str(sid) for sid in valid_segments_set}
            
            # Pre-compute active incoming connections to speed up the check
            active_incoming = set()
            for sid_str in valid_str_set:
                if sid_str in adjacency:
                    adj = adjacency[sid_str]
                    out_targets = adj.get('to', []) + adj.get('turns_into', []) + \
                                  adj.get('merges_into', []) + adj.get('u_turns_into', [])
                    for tgt in out_targets:
                        if str(tgt) in valid_str_set:
                            active_incoming.add(str(tgt))

            for sid in valid_segments_set:
                sid_str = str(sid)
                is_connected = False
                
                # Check outgoing and crossing relations
                if sid_str in adjacency:
                    adj = adjacency[sid_str]
                    out_targets = adj.get('to', []) + adj.get('turns_into', []) + \
                                  adj.get('merges_into', []) + adj.get('u_turns_into', []) + \
                                  adj.get('crosses', []) + adj.get('crossed_by', [])
                    
                    if any(str(tgt) in valid_str_set for tgt in out_targets):
                        is_connected = True
                
                # Check incoming relations (if another valid segment points to this one)
                if not is_connected and sid_str in active_incoming:
                    is_connected = True
                    
                if not is_connected:
                    logger.info(f"Segment {sid} is topologically isolated and will be removed.")
                    isolated_topo_segments.add(sid)
                    
            if not isolated_topo_segments:
                break
                
            valid_segments_set -= isolated_topo_segments
            logger.info(f"Removed {len(isolated_topo_segments)} topologically isolated segments.")

    valid_segments = list(valid_segments_set)
    
    df = df[df['segment_id'].isin(valid_segments)].reset_index(drop=True)
    
    # Export filtered road network
    filtered_network_path = os.path.join(output_dir, 'osm_network.gpkg')
    logger.info(f"Exporting filtered road network to {filtered_network_path}...")
    # Ensure IDs are compared as strings to match valid_segments list
    filtered_edges = edges_gdf[edges_gdf['segment_id'].astype(str).isin(valid_segments)]
    filtered_edges.to_file(filtered_network_path, driver='GPKG')

    # Filter and export controllers.json
    controllers_path = os.path.join(os.path.dirname(network_path) or '.', 'controllers.json')
    if os.path.exists(controllers_path):
        with open(controllers_path, 'r') as f:
            controllers = json.load(f)
            
        filtered_controllers = {}
        for cid, ctrl in controllers.items():
            valid_approaches = [s for s in ctrl.get('approach_segments', []) if str(s) in valid_segments_set]
            if not valid_approaches:
                continue
                
            ctrl['approach_segments'] = valid_approaches
            ctrl['junction_segments'] = [s for s in ctrl.get('junction_segments', []) if str(s) in valid_segments_set]
            filtered_controllers[cid] = ctrl
            
        filtered_controllers_path = os.path.join(output_dir, 'controllers.json')
        logger.info(f"Exporting filtered controllers ({len(filtered_controllers)}/{len(controllers)}) to {filtered_controllers_path}...")
        with open(filtered_controllers_path, 'w') as f:
            json.dump(filtered_controllers, f, indent=4)

    df, detected_lanes = calculate_signed_distance_and_lanes(df, edges_gdf, output_dir, config, logger, test=args.test)
    empirical_speeds = calculate_empirical_speeds(df, edges_gdf, output_dir, config, logger)
    
    # Export matched trajectories for feature extraction pipeline
    matched_trajectories_path = os.path.join(output_dir, "matched_trajectories.csv")
    logger.info(f"Exporting matched trajectories to {matched_trajectories_path}...")
    df.to_csv(matched_trajectories_path, index=False)
    
    logger.info(f"\nFinished mapping processing for {input_file}! Saved output to {output_dir}")
    logger.info(f"Execution for this file took {time.time() - file_start_time:.2f} seconds.")
    global_logger.info(f"Finished processing {input_file} in {time.time() - file_start_time:.2f} seconds.")
    return detected_lanes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Path to a single pNEUMA CSV file or directory")
    parser.add_argument('--test', action='store_true', help="Test/Debugging mode (Only processes a percentage vehicles)")
    args = parser.parse_args()
    
    overall_start = time.time()
    
    with open('removed_segments.json', 'r') as f:
        removed_segments_list = json.load(f)
    with open('segment_lanes.json', 'r') as f:
        segment_lanes_dict = json.load(f)
        
    segment_boundaries_dict = {}
    if os.path.exists('segment_boundaries.json'):
        with open('segment_boundaries.json', 'r') as f:
            segment_boundaries_dict = json.load(f)
            
    topological_adjacency_dict = {}
    if os.path.exists('topological_adjacency.json'):
        with open('topological_adjacency.json', 'r') as f:
            topological_adjacency_dict = json.load(f)
    elif os.path.exists('topological_adjacency_merged.json'):
        with open('topological_adjacency_merged.json', 'r') as f:
            topological_adjacency_dict = json.load(f)

    config = {
        "sampling_interval": 1000,
        "test_percentage": 0.05,
        "debug_segments": ["2423","1822","69", "2461"],
        "removed_segments": removed_segments_list,
        "segment_lanes": segment_lanes_dict,
        "segment_boundaries": segment_boundaries_dict,
        "topological_adjacency": topological_adjacency_dict,
        "map_matching_max_dist_start": 5,
        "map_matching_max_dist_end": 50,
        "map_matching_step": 5,
        "rel_heading_limit": 90.0,
        "partial_traversal_length_thresh": 50.0,
        "partial_traversal_prop_thresh": 0.6,
        "min_vehicles_per_segment": 5,
        "jenks_min_points": 50,
        "jenks_speed_thresholds": [1.0],
        "jenks_heading_threshold": 5.0,
        "jenks_target_per_bin": 10,
        "jenks_max_target_calc": 15000,
        "jenks_density_power": 1.0,
        "jenks_bins": 40,
        "avg_lane_width": 3.2,
        "min_lane_width_loose": 1.6,
        "min_lane_width_strict": 1.75,
        "max_lane_width": 4.5,
        "default_speeds": {'primary': 14.0, 'secondary': 11.0, 'tertiary': 9.0, 'trunk': 20.0, 'residential': 8.0, 'unclassified': 8.0},
        "default_speed_fallback": 10.0,
        "signal_proximity_threshold": 5.0
    }
    
    os.makedirs("processed_data", exist_ok=True)
    global_log_file = os.path.join("processed_data", "map-matching.log")
    global_logger = logging.getLogger("global_logger")
    global_logger.setLevel(logging.INFO)
    if global_logger.hasHandlers():
        global_logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    gfh = logging.FileHandler(global_log_file, mode='w', encoding='utf-8')
    gfh.setFormatter(formatter)
    global_logger.addHandler(gfh)
    gch = logging.StreamHandler()
    gch.setFormatter(formatter)
    global_logger.addHandler(gch)
    
    global_logger.info("Pipeline started.")

    if os.path.isdir(args.input_path):
        files = sorted([os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.csv')])
    else:
        files = [args.input_path]
        
    if not files:
        global_logger.info("No CSV file(s) found.")
        return
        
    global_logger.info("Loading OSM network...")
    network_path = 'osm_network.gpkg'
    edges_gdf = gpd.read_file(network_path)
    
    all_detected_lanes = {}
    for input_file in files:
        file_detected_lanes = process_single_file(input_file, edges_gdf, network_path, args, config, global_logger)
        if file_detected_lanes:
            for seg_id, count in file_detected_lanes.items():
                if seg_id not in all_detected_lanes:
                    all_detected_lanes[seg_id] = []
                all_detected_lanes[seg_id].append(count)

    osm_lanes = {}
    for row in edges_gdf.itertuples():
        seg_id = str(row.segment_id)
        try:
            if pd.isna(row.lanes):
                lanes = None
            else:
                lanes = int(row.lanes)
        except (ValueError, TypeError):
            lanes = None
        osm_lanes[seg_id] = lanes
        
    segment_lane_features = {}
    average_detected_lanes = {}
    for seg_id, counts in all_detected_lanes.items():
        avg_lanes = round(sum(counts) / len(counts))
        average_detected_lanes[seg_id] = avg_lanes
        segment_lane_features[seg_id] = {
            "average_detected_lanes": avg_lanes,
            "detected_lanes_list": counts,
            "osm_defined_lanes": osm_lanes.get(seg_id)
        }
        
    avg_lanes_file = os.path.join("processed_data", "average_detected_lanes.json")
    with open(avg_lanes_file, "w") as f:
        json.dump(segment_lane_features, f, indent=4)
    global_logger.info(f"Saved average detected lanes to {avg_lanes_file}")
    
    mismatch_log_file = os.path.join("processed_data", "lane-mismatch.log")
    mismatches_found = 0
    with open(mismatch_log_file, "w") as f:
        for seg_id, avg_lanes in average_detected_lanes.items():
            osm_count = osm_lanes.get(seg_id)
            if avg_lanes != osm_count:
                f.write(f"Segment {seg_id}: Average Detected = {avg_lanes}, OSM Defined = {osm_count}\n")
                mismatches_found += 1
                
    global_logger.info(f"Found {mismatches_found} segments with lane mismatches. See {mismatch_log_file}")

    global_logger.info(f"Total mapping execution completed in {time.time() - overall_start:.2f} seconds.")

if __name__ == '__main__':
    main()