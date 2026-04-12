import argparse
import time
import os
import shutil
import json
import multiprocessing
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jenkspy
from shapely.geometry import Point, LineString
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap

def get_osm_map(edges_gdf):
    map_con = InMemMap("osm", use_latlon=False, use_rtree=True, index_edges=True)
    for _, row in edges_gdf.iterrows():
        u, v = int(row['u']), int(row['v'])
        coords = list(row['geometry'].coords)
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
                
    return data_list

def parse_pneuma_to_long(filepath, config, test=False):
    print(f"Parsing {filepath}...")
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
        print(f"Test mode: taking {config['test_percentage']*100:.1f}% of non-motorcycle vehicles ({limit} out of {num_vehicles})")
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

    data_list = []
    num_processes = multiprocessing.cpu_count()
    
    # Use imap_unordered to process chunks as they are generated and yielded
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(_parse_pneuma_chunk, chunk_generator()):
            data_list.extend(result)

    df = pd.DataFrame(data_list, columns=[
        'track_id', 'type', 'traveled_d', 'avg_speed', 
        'lat', 'lon', 'speed', 'lon_acc', 'lat_acc', 'time'
    ])
    
    print(f"Parsing took {time.time() - start_time:.2f} seconds. Parsed {len(df)} points.")
    return df

# Global variables for worker processes to prevent repeated memory allocation
global_map_con = None
global_edge_data = None
global_lanes_map = None
global_len_map = None
global_uv_to_seg = None
global_hw_map = None
global_config = None

def _map_match_worker_init(gpkg_path, config):
    global global_map_con, global_edge_data, global_lanes_map, global_len_map, global_uv_to_seg, global_hw_map, global_config
    global_config = config
    
    edges_gdf = gpd.read_file(gpkg_path)
    global_map_con = get_osm_map(edges_gdf)
    
    global_edge_data = {}
    global_lanes_map = {}
    global_len_map = {}
    global_uv_to_seg = {}
    global_hw_map = {}
    for _, row in edges_gdf.iterrows():
        key = (int(row['u']), int(row['v']))
        global_lanes_map[key] = int(row['lanes'])
        global_len_map[key] = float(row['length'])
        global_uv_to_seg[key] = str(row['segment_id'])
        global_hw_map[key] = str(row['highway'])
        
        coords = np.array(row['geometry'].coords)
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
    
    for key, indices in edge_groups.items():
        if key not in global_edge_data:
            for idx in indices:
                segment_ids[idx] = ""
                highways_list[idx] = ""
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
        
        for i, idx in enumerate(indices):
            signed_distances[idx] = signed_d[i]
            segment_ids[idx] = seg_id
            num_lanes_list[idx] = n_lanes
            segment_lengths[idx] = total_l
            prop_distances[idx] = dist_along[i] / total_l
            highways_list[idx] = hw
            t_projs[idx] = dist_along[i]
            rel_headings[idx] = rel_h[i]
            
    group_results['signed_dist'] = signed_distances
    group_results['segment_id'] = segment_ids
    group_results['num_lanes'] = num_lanes_list
    group_results['segment_length'] = segment_lengths
    group_results['prop_dist'] = prop_distances
    group_results['rel_heading'] = rel_headings
    group_results['highway'] = highways_list
    group_results['t_proj'] = t_projs
    
    group_results = group_results[group_results['rel_heading'] <= global_config["rel_heading_limit"]].copy()
    
    if group_results.empty:
        return pd.DataFrame()
        
    def is_fully_traversed(grp):
        if grp['segment_length'].iloc[0] < global_config["partial_traversal_length_thresh"]:
            return True
        return (grp['prop_dist'].max() - grp['prop_dist'].min()) >= global_config["partial_traversal_prop_thresh"]

    valid_traversals = group_results.groupby('segment_id').filter(is_fully_traversed)
    return valid_traversals

def perform_map_matching(df, crs, gpkg_path, config):
    print("Performing map matching and distance calculation with pool load balancing...")
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
    print(f"Map matching and distance calculation took {time.time() - start_time:.2f} seconds. Matched {len(final_df)} points.")
    return final_df

def calculate_signed_distance_and_lanes(df, edges_gdf, output_dir, config, test=False):
    if test:
        print("Performing Jenks Optimization...")
    start_time = time.time()

    # Calculate and print max samples
    if test:
        counts = df[df['segment_id'] != ""]['segment_id'].value_counts()
        if not counts.empty:
            print(f"Maximum samples in a single segment: {counts.max()}")

    # Process Jenks
    lane_boundaries = {}
    jenks_segments_used = []
    failed_segments = []

    grouped = df.groupby('segment_id')

    for seg_id, group in grouped:
        if seg_id == "": continue

        d_vals = group['signed_dist'].values
        # Use 98th and 2nd percentiles for robust road width estimation
        abs_max_d = np.percentile(d_vals, 98)
        abs_min_d = np.percentile(d_vals, 2)
        road_width = abs_max_d - abs_min_d

        # D = Right_Edge - Vehicle_Distance. Clip extreme outliers on both sides.
        df.loc[group.index, 'D'] = (abs_max_d - group['signed_dist']).clip(0.0, road_width)
        D_vals = df.loc[group.index, 'D'].values

        num_lanes_osm = int(group['num_lanes'].iloc[0])

        if len(D_vals) > config["jenks_min_points"]:
            # Iterative filtering for speed (5, 3, 1) while maintaining the current rel_heading filter
            hq_D_final = None
            for s_thresh in config["jenks_speed_thresholds"]:
                mask = (group['speed'] > s_thresh) & (group['rel_heading'] < config["jenks_heading_threshold"])
                if mask.sum() >= config["jenks_min_points"]:
                    hq_D_final = D_vals[mask.values]
                    if test:
                        print(f"Segment {seg_id}: Retained {len(hq_D_final)} points using speed > {s_thresh}")
                    break
            
            if hq_D_final is None:
                hq_D_final = D_vals
                if test:
                    print(f"Segment {seg_id}: Defaulting to original distribution")

            # Use raw min/max instead of percentiles
            road_min, road_max = hq_D_final.min(), hq_D_final.max()
            hq_D_trimmed = hq_D_final # No longer trimming

            # Balanced Sampling: Take an equal-ish number of samples from across the distribution
            counts, bins = np.histogram(hq_D_trimmed, bins=config["jenks_bins"])
            target_per_bin = max(config["jenks_target_per_bin"], config["jenks_max_target_calc"] // (len(counts[counts > 0]) or 1))

            balanced_sample = []
            for i in range(len(counts)):
                if counts[i] == 0: continue
                bin_mask = (hq_D_trimmed >= bins[i]) & (hq_D_trimmed <= bins[i+1])
                bin_vals = hq_D_trimmed[bin_mask]
                balanced_sample.extend(np.random.choice(bin_vals, min(len(bin_vals), target_per_bin), replace=False))

            sample = np.array(balanced_sample)

            # Iterative Lane Partitioning: Start with max_distance / avg_lane_width
            max_distance = (road_max - road_min)  
            avg_lane_width = config["avg_lane_width"]
            initial_k = max(1, round(max_distance / avg_lane_width))

            best_breaks = None
            if seg_id in config["fixed_segments"]:
                k = config["fixed_segments"][seg_id]
                if k == 1:
                    best_breaks = [road_min, road_max]
                    if test:
                        print(f"Segment {seg_id}: Manual override applied. Assigned 1 lane [road_min, road_max].")
                else:
                    try:
                        best_breaks = jenkspy.jenks_breaks(sample, n_classes=k)
                        if test:
                            print(f"Segment {seg_id}: Manual override applied. Assigned {k} lanes using Jenks.")
                    except ValueError:
                        best_breaks = [road_min + i * (max_distance / k) for i in range(k + 1)]
                        failed_segments.append(seg_id)
                        if test:
                            print(f"Segment {seg_id}: Manual override applied. Jenks failed, using equal widths for {k} lanes.")
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
                                print(f"Segment {seg_id}: Increased 1-lane guess to {k} lanes based on Jenks.")
                            break
                    except ValueError:
                        continue
                
                # If neither 2 nor 3 lanes worked, default to road_min and road_max
                if best_breaks is None:
                    best_breaks = [road_min, road_max]
                    if test:
                        print(f"Segment {seg_id}: Defaulted to 1 lane [road_min, road_max].")
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
                                print(f"Segment {seg_id}: Determined {k} lanes (Initial guess: {initial_k}).")
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
                                    print(f"Segment {seg_id}: Reduced to {k_red} lanes (Initial guess: {initial_k}).")
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
                        print(f"Segment {seg_id}: Jenks failed to find valid breaks. Used initial guess with {initial_k} lanes.")
                except ValueError:
                    lane_boundaries[seg_id] = [road_min + i * avg_lane_width for i in range(initial_k + 1)]
        else:
            if seg_id in config["fixed_segments"]:
                num_lanes_fixed = config["fixed_segments"][seg_id]
                lane_boundaries[seg_id] = [i * config["avg_lane_width"] for i in range(num_lanes_fixed + 1)]
                if test:
                    print(f"Segment {seg_id}: Manual override applied. Assigned {num_lanes_fixed} lanes (low sample fallback).")
            else:
                lane_boundaries[seg_id] = [i * config["avg_lane_width"] for i in range(num_lanes_osm + 1)]

    # Lane Assignment and Outlier Removal
    df['lane_index'] = 0
    df['is_outlier'] = False
    
    lanes_updated_count = 0
    for seg_id, bounds in lane_boundaries.items():
        mask = df['segment_id'] == seg_id
        if not mask.any(): continue
            
        raw_indices = np.digitize(df.loc[mask, 'D'], bins=bounds)
        
        # Identify outliers (indices 0 for < bounds[0] or len(bounds) for >= bounds[-1])
        seg_outliers = (raw_indices == 0) | (raw_indices == len(bounds))
        df.loc[df[mask].index[seg_outliers], 'is_outlier'] = True
        
        # Map valid points to 0-based lane indices (clipping here just avoids OutOfBounds errors before removal)
        df.loc[mask, 'lane_index'] = np.clip(raw_indices, 1, len(bounds) - 1) - 1
        
        # Update num_lanes to match the number of detected Jenks lanes (or the original OSM lanes if Jenks wasn't used)
        old_lanes = df.loc[mask, 'num_lanes'].iloc[0]
        new_lanes = len(bounds) - 1
        if old_lanes != new_lanes:
            lanes_updated_count += 1
            
        df.loc[mask, 'num_lanes'] = new_lanes

    if lanes_updated_count > 0:
        print(f"Successfully updated 'num_lanes' on {lanes_updated_count} segments using empirical Jenks data.")
    else:
        print("No segments had their 'num_lanes' altered from the OSM default (likely due to low sample size or Jenks agreeing with OSM).")

    flagged_outliers = df['is_outlier'].sum()
    print(f"Flagged {flagged_outliers} outlier trajectory points that did not fit into edge lanes.")

    print(f"Jenks Optimization took {time.time() - start_time:.2f} seconds.")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'lane_boundaries.json'), 'w') as f:
        json.dump(lane_boundaries, f, indent=4)        
    
    # Plot debug histograms for specific segments
    debug_dir = os.path.join(output_dir, 'debugged_segments')
    failed_dir = os.path.join(debug_dir, 'failed_segments')
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    segments_to_plot = set(config["debug_segments"]).union(set(failed_segments))
    print(f"Number of failed segments to plot: {len(failed_segments)}.")
    
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
            print(f"Generated debug histogram for segment {seg}.")
        
    return df

def calculate_empirical_speeds(df, edges_gdf, output_dir, config):
    print("Calculating empirical speeds...")
    start_time = time.time()
    
    speed_df = df
    
    # Calculate traversal speeds (simply 95th percentile of observed speeds per segment)
    speeds = speed_df.groupby('segment_id')['speed'].quantile(0.95).to_dict()
    
    # Impute missing
    # Default highway speeds in m/s
    default_speeds = config["default_speeds"]
    
    empirical_speeds = {}
    for _, row in edges_gdf.iterrows():
        seg_id = str(row['segment_id'])
        hw = row['highway']
        if seg_id in speeds:
            empirical_speeds[seg_id] = speeds[seg_id]
        else:
            empirical_speeds[seg_id] = default_speeds.get(hw, config["default_speed_fallback"])
            
    with open(os.path.join(output_dir, 'empirical_free_flow_speeds.json'), 'w') as f:
        json.dump(empirical_speeds, f, indent=4)
        
    print(f"Speed calculation took {time.time() - start_time:.2f} seconds.")
    return empirical_speeds

def process_single_file(input_file, edges_gdf, network_path, args, config):
    print(f"\n--- Processing {input_file} ---")
    file_start_time = time.time()
    
    df = parse_pneuma_to_long(input_file, config, test=args.test)
    if df.empty:
        print("No data parsed.")
        return
        
    df = perform_map_matching(df, edges_gdf.crs, network_path, config)
    if df.empty:
        return
        
    # Filter segments with fewer than 10 unique vehicles or in removed_segments
    print(f"Filtering segments (Min {config['min_vehicles_per_segment']} vehicles, Exclude blacklist: {len(config['removed_segments'])})...")
    seg_counts = df[df['segment_id'] != ""].groupby('segment_id')['track_id'].nunique()
    
    # Identify segments that meet the volume threshold AND are not in the blacklist
    valid_segments = [s for s in seg_counts.index if s not in config["removed_segments"] and seg_counts[s] >= config["min_vehicles_per_segment"]]
    
    initial_count = len(df)
    df = df[df['segment_id'].isin(valid_segments)].reset_index(drop=True)
    
    num_removed_segs = len(seg_counts) - len(valid_segments)
    print(f"Removed {initial_count - len(df)} points across {num_removed_segs} segments (low volume or blacklisted).")

    file_name = os.path.splitext(os.path.basename(input_file))[0]
    date_part = file_name.split('_')[0]
    output_dir = os.path.join("processed_data", date_part, file_name)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Export filtered road network
    filtered_network_path = os.path.join(output_dir, 'osm_network.gpkg')
    print(f"Exporting filtered road network to {filtered_network_path}...")
    # Ensure IDs are compared as strings to match valid_segments list
    filtered_edges = edges_gdf[edges_gdf['segment_id'].astype(str).isin(valid_segments)]
    filtered_edges.to_file(filtered_network_path, driver='GPKG')

    df = calculate_signed_distance_and_lanes(df, edges_gdf, output_dir, config, test=args.test)
    empirical_speeds = calculate_empirical_speeds(df, edges_gdf, output_dir, config)
    
    # Export matched trajectories for feature extraction pipeline
    matched_trajectories_path = os.path.join(output_dir, "matched_trajectories.csv")
    print(f"Exporting matched trajectories to {matched_trajectories_path}...")
    df.to_csv(matched_trajectories_path, index=False)
    
    print(f"\nFinished mapping processing for {input_file}! Saved output to {output_dir}")
    print(f"Execution for this file took {time.time() - file_start_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Path to a single pNEUMA CSV file or directory")
    parser.add_argument('--test', action='store_true', help="Test/Debugging mode (Only processes a percentage vehicles)")
    args = parser.parse_args()
    
    overall_start = time.time()
    
    with open('removed_segments.json', 'r') as f:
        removed_segments_list = json.load(f)
    with open('fixed_segments.json', 'r') as f:
        fixed_segments_dict = json.load(f)

    config = {
        "sampling_interval": 1000,
        "test_percentage": 0.05,
        "debug_segments": ["1750", "1909", "1751", "1756", "1753", "1834", "2227", "1836","208","220"],
        "removed_segments": removed_segments_list,
        "fixed_segments": fixed_segments_dict,
        "map_matching_max_dist_start": 5,
        "map_matching_max_dist_end": 50,
        "map_matching_step": 5,
        "rel_heading_limit": 90.0,
        "partial_traversal_length_thresh": 50.0,
        "partial_traversal_prop_thresh": 0.6,
        "min_vehicles_per_segment": 5,
        "jenks_min_points": 50,
        "jenks_speed_thresholds": [7.5, 5.0, 3.0, 1.0],
        "jenks_heading_threshold": 5.0,
        "jenks_target_per_bin": 10,
        "jenks_max_target_calc": 15000,
        "jenks_bins": 40,
        "avg_lane_width": 3.2,
        "min_lane_width_loose": 1.6,
        "min_lane_width_strict": 1.75,
        "max_lane_width": 4.5,
        "default_speeds": {'primary': 14.0, 'secondary': 11.0, 'tertiary': 9.0, 'trunk': 20.0, 'residential': 8.0, 'unclassified': 8.0},
        "default_speed_fallback": 10.0
    }
    
    if os.path.isdir(args.input_path):
        files = sorted([os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.csv')])
    else:
        files = [args.input_path]
        
    if not files:
        print("No CSV file(s) found.")
        return
        
    print("Loading OSM network...")
    network_path = 'osm_network_merged.gpkg'
    edges_gdf = gpd.read_file(network_path)
    
    for input_file in files:
        process_single_file(input_file, edges_gdf, network_path, args, config)

    print(f"\nTotal mapping execution completed in {time.time() - overall_start:.2f} seconds.")

if __name__ == '__main__':
    main()