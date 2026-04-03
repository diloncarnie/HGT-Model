import argparse
import time
import os
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
from scipy.spatial import KDTree

sampling_interval = 1000  # 1.0Hz interpolation
test_percentage = 0.05  # Percentage of vehicles to take in test mode
debug_segments = ["1750", "1909", "1751", "165", "1756", "1753", "2229", "12974", "335", "1895", "1834", "10863"]
removed_vehicles = []
removed_segments = ["1770","1797","1784","1800","14486","1835","1747","1904","1905","1917","1897","1757","1794",
                    "2389","337","287","329","455","10955","10861","10854", "2390","12601","81","82","338","160","2248","14465","14389","10880","256","260","190","12976","286","82","159","161","160","159","214","286","250","249"]

fixed_segments = {  "0":3,"3":3,
                    "165":2,"177":3,"199":3,"243":2,"299":3,"291":3,"293":3,"296":3,"399":3,"210":2,"211":3,"226":3,"240":2,"285":2,"290":3,"298":4,"300":4,"325":3,"335":2,"389":3,
                    "1801":5,"1802":3,"1803":3,
                    "2228":3, "2229":3, "2250":4,
                    "1748":3,"1750":4,"1751":3,"1753":3,"1756":3,"1759":3,"1760":5,"1764":3,"1766":4,"1768":4,"1769":5,"1771":3,"1772":3,"1793":4,"1798":3,"1799":3,
                    "1804":3,"1828": 5,"1834": 4,"1836":4,"1837":5,"1839":5,"1848":4,"1849":3,"1851":3,"1855":3,"1895":3,"1896":3,"1899":3,
                    "1902":3, "1903":3, "1906":4, "1908":2, "1909":4,
                    "6579":4,
                    "10869":3,"10863":3,"10855":3,"10858":3,"10862":3,"10865":3,"10872":4,"10874":3,"10882":3,"10956":5,
                    "12974":4,"12975":3,
                    "13064":4,"13232":4,"13803":5,"13869":3,"13871":3,
                    "14317":3,"14346":3}

def get_osm_map(edges_gdf):
    map_con = InMemMap("osm", use_latlon=False, use_rtree=True, index_edges=True)
    for _, row in edges_gdf.iterrows():
        u, v = int(row['u']), int(row['v'])
        coords = list(row['geometry'].coords)
        map_con.add_node(u, (coords[0][0], coords[0][1]))
        map_con.add_node(v, (coords[-1][0], coords[-1][1]))
        map_con.add_edge(u, v)
    return map_con

def parse_pneuma_to_long(filepath, test=False):
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
                    
        limit = max(1, int(num_vehicles * test_percentage))
        print(f"Test mode: taking {test_percentage*100:.1f}% of non-motorcycle vehicles ({limit} out of {num_vehicles})")
    else:
        limit = float('inf')

    data_list = []
    with open(filepath, 'r') as f:
        # Skip header if present
        first_line = f.readline()
        if "track_id" not in first_line:
            f.seek(0)
            
        count = 0
        for line in f:
            if count >= limit:
                break
            
            parts = line.strip().split(';')
            if len(parts) < 4: continue
            
            track_id = parts[0].strip()
            v_type = parts[1].strip()
            
            if track_id in removed_vehicles:
                continue
      
            
            if v_type == 'Motorcycle':
                continue
                
            traveled_d = parts[2].strip()
            avg_speed_kmh = float(parts[3].strip())
            avg_speed_ms = avg_speed_kmh / 3.6
            
            points = parts[4:]
            sampled_pts = []
            last_included_timestamp = None
            variance = int(sampling_interval * 0.1)  # 10% variance
            for i in range(0, len(points)-1, 6):
                current_gap = sampling_interval + np.random.randint(-variance, variance + 1)
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
                
            count += 1
            
    df = pd.DataFrame(data_list, columns=[
        'track_id', 'type', 'traveled_d', 'avg_speed', 
        'lat', 'lon', 'speed', 'lon_acc', 'lat_acc', 'time'
    ])
    
    print(f"Parsing took {time.time() - start_time:.2f} seconds. Parsed {len(df)} points.")
    return df

def process_map_matching_chunk(chunk_df, crs, gpkg_path='osm_network.gpkg', max_dist_start=5, max_dist_end=15, step=5):
    edges_gdf = gpd.read_file(gpkg_path)
    map_con = get_osm_map(edges_gdf)
    
    # Pre-map edge attributes for fast lookup
    edge_map = {}
    lanes_map = {}
    len_map = {}
    uv_to_seg = {}
    hw_map = {}
    for _, row in edges_gdf.iterrows():
        key = (int(row['u']), int(row['v']))
        edge_map[key] = row['geometry']
        lanes_map[key] = int(row['lanes'])
        len_map[key] = float(row['length'])
        uv_to_seg[key] = str(row['segment_id'])
        hw_map[key] = str(row['highway'])
    
    matched_results = []
    grouped = chunk_df.groupby('track_id')
    
    for track_id, group in grouped:
        path = group[['x', 'y']].values.tolist()
        
        matcher = None
        matched_states = None
        for max_dist in range(max_dist_start, max_dist_end + 1, step):
            # non_emitting_states=False ensures 1:1 mapping between observations and states
            matcher = DistanceMatcher(map_con, max_dist=max_dist, obs_noise=max_dist, non_emitting_states=False)
            states, last_idx = matcher.match(path)
            if states and (last_idx == len(path) - 1 or max_dist == max_dist_end):
                matched_states = states
                break
                
        if not matched_states:
            continue
            
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
        
        # Calculate vehicle azimuth (azcar)
        dx = group_results['x'].diff().bfill().fillna(0)
        dy = group_results['y'].diff().bfill().fillna(0)
        group_results['azcar'] = np.arctan2(dx, dy) * 180 / np.pi
        
        # Calculate signed distance and segment info immediately
        signed_distances = []
        segment_ids = []
        num_lanes_list = []
        segment_lengths = []
        prop_distances = []
        rel_headings = []
        highways_list = []
        t_projs = []
        
        for idx, row in group_results.iterrows():
            u, v = row['matched_u'], row['matched_v']
            key = (u, v)
            if key not in edge_map:
                signed_distances.append(0.0)
                segment_ids.append("")
                num_lanes_list.append(0)
                segment_lengths.append(0.0)
                prop_distances.append(0.5)
                rel_headings.append(0.0)
                highways_list.append("")
                t_projs.append(0.0)
                continue
                
            geom = edge_map[key]
            coords = list(geom.coords)
            px, py = row['x'], row['y']
            
            best_dist = float('inf')
            signed_d = 0.0
            dist_along = 0.0
            best_az_sub = 0.0
            
            current_accumulated_l = 0.0
            for i in range(len(coords)-1):
                ax, ay = coords[i]
                bx, by = coords[i+1]
                a = np.array([ax, ay])
                b = np.array([bx, by])
                p = np.array([px, py])
                
                ab = b - a
                ap = p - a
                sub_l2 = np.dot(ab, ab)
                if sub_l2 == 0: continue
                
                t = max(0, min(1, np.dot(ap, ab) / sub_l2))
                proj = a + t * ab
                dist = np.linalg.norm(p - proj)
                
                if dist < best_dist:
                    best_dist = dist
                    cross = ab[0]*ap[1] - ab[1]*ap[0]
                    signed_d = dist * (-1 if cross > 0 else 1)
                    dist_along = current_accumulated_l + t * np.sqrt(sub_l2)
                    # Local azimuth of this specific sub-segment
                    best_az_sub = np.arctan2(ab[0], ab[1]) * 180 / np.pi
                
                current_accumulated_l += np.sqrt(sub_l2)
            
            total_l = len_map.get(key, current_accumulated_l or 1.0)
            signed_distances.append(signed_d)
            segment_ids.append(uv_to_seg.get(key, str(key)))
            num_lanes_list.append(lanes_map.get(key, 2))
            segment_lengths.append(total_l)
            prop_distances.append(dist_along / total_l)
            highways_list.append(hw_map.get(key, "unknown"))
            t_projs.append(dist_along)
            
            # Relative heading: difference between vehicle azimuth and sub-segment azimuth
            rel_h = np.abs((row['azcar'] - best_az_sub + 180) % 360 - 180)
            rel_headings.append(rel_h)
            
        group_results['signed_dist'] = signed_distances
        group_results['segment_id'] = segment_ids
        group_results['num_lanes'] = num_lanes_list
        group_results['segment_length'] = segment_lengths
        group_results['prop_dist'] = prop_distances
        group_results['rel_heading'] = rel_headings
        group_results['highway'] = highways_list
        group_results['t_proj'] = t_projs
        
        # Filter out points with relative heading > 45 degrees
        group_results = group_results[group_results['rel_heading'] <= 45.0].copy()
        
        if group_results.empty:
            continue
            
        # Filter out partial traversals (traversed < 80% of the segment length)
        # Skip this filter for segments shorter than 25 meters to avoid pruning fast traversals
        def is_fully_traversed(group):
            if group['segment_length'].iloc[0] < 30.0:
                return True
            return (group['prop_dist'].max() - group['prop_dist'].min()) >= 0.75

        valid_traversals = group_results.groupby('segment_id').filter(is_fully_traversed)
        group_results = valid_traversals.copy()
        
        if group_results.empty:
            continue
                
        matched_results.append(group_results)
        
    if not matched_results:
        return pd.DataFrame()
        
    return pd.concat(matched_results, ignore_index=True)

def perform_map_matching(df, crs, gpkg_path='osm_network.gpkg', max_dist_start=5, max_dist_end=25, step=5):
    print("Performing iterative map matching and distance calculation with multiprocessing...")
    start_time = time.time()
    
    # Project to CRS
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    gdf = gdf.to_crs(crs)
    df['x'] = gdf.geometry.x
    df['y'] = gdf.geometry.y
    
    track_ids = df['track_id'].unique()
    num_processes = multiprocessing.cpu_count()
    chunks = np.array_split(track_ids, num_processes)
    
    chunk_dfs = [df[df['track_id'].isin(chunk.tolist())] for chunk in chunks if len(chunk) > 0]
    
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.starmap(process_map_matching_chunk, [(cdf, crs, gpkg_path, max_dist_start, max_dist_end, step) for cdf in chunk_dfs])
    pool.close()
    pool.join()
    
    final_df = pd.concat([r for r in results if not r.empty], ignore_index=True) if results else pd.DataFrame()
    print(f"Map matching and distance calculation took {time.time() - start_time:.2f} seconds. Matched {len(final_df)} points.")
    return final_df

def calculate_signed_distance_and_lanes(df, edges_gdf, output_dir, test=False):
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

        if len(D_vals) > 50:
            # Iterative filtering for speed (5, 3, 1) while maintaining the current rel_heading filter
            hq_D_final = None
            for s_thresh in [7.5, 5.0, 3.0, 1.0]:
                mask = (group['speed'] > s_thresh) & (group['rel_heading'] < 5.0)
                if mask.sum() >= 50:
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
            counts, bins = np.histogram(hq_D_trimmed, bins=40)
            target_per_bin = max(10, 15000 // (len(counts[counts > 0]) or 1))

            balanced_sample = []
            for i in range(len(counts)):
                if counts[i] == 0: continue
                bin_mask = (hq_D_trimmed >= bins[i]) & (hq_D_trimmed <= bins[i+1])
                bin_vals = hq_D_trimmed[bin_mask]
                balanced_sample.extend(np.random.choice(bin_vals, min(len(bin_vals), target_per_bin), replace=False))

            sample = np.array(balanced_sample)

            # Iterative Lane Partitioning: Start with max_distance / avg_lane_width
            max_distance = (road_max - road_min)  
            avg_lane_width = 3.2
            initial_k = max(1, round(max_distance / avg_lane_width))

            best_breaks = None
            if seg_id in fixed_segments:
                k = fixed_segments[seg_id]
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
                            if width < 1.6 or width > 4.5:
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
                            if width < 1.75 or width > 4.5:
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
                                if width < 1.75 or width > 4.5:
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
            if seg_id in fixed_segments:
                num_lanes_fixed = fixed_segments[seg_id]
                lane_boundaries[seg_id] = [i * 3.2 for i in range(num_lanes_fixed + 1)]
                if test:
                    print(f"Segment {seg_id}: Manual override applied. Assigned {num_lanes_fixed} lanes (low sample fallback).")
            else:
                lane_boundaries[seg_id] = [i * 3.2 for i in range(num_lanes_osm + 1)]

    # Lane Assignment and Outlier Removal
    df['lane_index'] = 0
    outlier_mask = pd.Series(False, index=df.index)
    
    lanes_updated_count = 0
    for seg_id, bounds in lane_boundaries.items():
        mask = df['segment_id'] == seg_id
        if not mask.any(): continue
            
        raw_indices = np.digitize(df.loc[mask, 'D'], bins=bounds)
        
        # Identify outliers (indices 0 for < bounds[0] or len(bounds) for >= bounds[-1])
        seg_outliers = (raw_indices == 0) | (raw_indices == len(bounds))
        outlier_mask.loc[df[mask].index[seg_outliers]] = True
        
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

    initial_len = len(df)
    df = df[~outlier_mask].reset_index(drop=True)
    removed_outliers = initial_len - len(df)
    print(f"Removed {removed_outliers} outlier trajectory points that did not fit into edge lanes.")

    print(f"Jenks Optimization took {time.time() - start_time:.2f} seconds.")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'lane_boundaries.json'), 'w') as f:
        json.dump(lane_boundaries, f, indent=4)        
    
    # Plot debug histograms for specific segments
    debug_dir = os.path.join(output_dir, 'debugged_segments')
    failed_dir = os.path.join(debug_dir, 'failed_segments')
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    segments_to_plot = set(debug_segments).union(set(failed_segments))
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

def calculate_empirical_speeds(df, edges_gdf, output_dir):
    print("Calculating empirical speeds...")
    start_time = time.time()
    
    speed_df = df[df['type'].isin(['Car', 'Medium Vehicle'])]
    
    # Calculate traversal speeds (simply 95th percentile of observed speeds per segment)
    speeds = speed_df.groupby('segment_id')['speed'].quantile(0.95).to_dict()
    
    # Impute missing
    # Default highway speeds in m/s
    default_speeds = {'primary': 14.0, 'secondary': 11.0, 'tertiary': 9.0, 'trunk': 20.0, 'residential': 8.0, 'unclassified': 8.0}
    
    empirical_speeds = {}
    for _, row in edges_gdf.iterrows():
        seg_id = str(row['segment_id'])
        hw = row['highway']
        if seg_id in speeds:
            empirical_speeds[seg_id] = speeds[seg_id]
        else:
            empirical_speeds[seg_id] = default_speeds.get(hw, 10.0)
            
    with open(os.path.join(output_dir, 'empirical_free_flow_speeds.json'), 'w') as f:
        json.dump(empirical_speeds, f, indent=4)
        
    print(f"Speed calculation took {time.time() - start_time:.2f} seconds.")
    return empirical_speeds

def process_frenet_for_timestamp(group_data_tuple):
    timestamp, df_group, adjacency, speeds = group_data_tuple
    
    if df_group.empty:
        return pd.DataFrame()
        
    # Build KDTree
    coords = df_group[['x', 'y']].values
    tree = KDTree(coords)
    
    cavs = df_group[df_group['is_CAV'] == True]
    if cavs.empty:
        return pd.DataFrame()
        
    results = []
    
    for idx, ego in cavs.iterrows():
        ego_idx = df_group.index.get_loc(idx)
        ego_pos = coords[ego_idx]
        ego_seg = ego['segment_id']
        ego_D = ego['D']
        ego_S = ego['t_proj'] # calculated explicitly in map matching
        ego_az = ego['azcar']
        
        neighbors_idx = tree.query_ball_point(ego_pos, r=70) # 50m max long + 20m lat
        
        valid_neighbors = []
        ego_adj = adjacency.get(ego_seg, {})
        successors = ego_adj.get('successors', [])
        predecessors = ego_adj.get('predecessors', [])
        successor_lengths = ego_adj.get('successor_lengths', [])
        predecessor_lengths = ego_adj.get('predecessor_lengths', [])

        for n_idx in neighbors_idx:
            if n_idx == ego_idx: continue
            n_row = df_group.iloc[n_idx]
            n_seg = n_row['segment_id']
            
            # Heading Filter: Use dynamic vehicle azimuth (azcar) instead of static segment headings
            n_az = n_row['azcar']
            heading_diff = abs((n_az - ego_az + 180) % 360 - 180)
            if heading_diff > 65.0:
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
            v_len = 5.0 if n['type'] in ['Car', 'Taxi'] else 12.5 # approx lengths
            
            if -50 <= dS <= 50:
                if -1.6 <= dD <= 1.6:
                    if dS > 0: zones['proceeding'].append((n['speed'], v_len))
                    elif dS < 0: zones['following'].append((n['speed'], v_len))
                elif dD > 1.6:
                    if dS >= 0: zones['leftwards_proceeding'].append((n['speed'], v_len))
                    else: zones['leftwards_following'].append((n['speed'], v_len))
                elif dD < -1.6:
                    if dS >= 0: zones['rightwards_proceeding'].append((n['speed'], v_len))
                    else: zones['rightwards_following'].append((n['speed'], v_len))
                
        ego_res = ego.to_dict()
        rem_lanes = max(1, ego['num_lanes'] - 1)
        
        for z in zones.keys():
            if not zones[z]:
                ego_res[f'raw_density_{z}'] = 0
                ego_res[f'raw_speed_{z}'] = speeds.get(ego_seg, 10.0) 
                ego_res[f'relative_occupancy_{z}'] = 0.0
                ego_res[f'relative_speed_{z}'] = 1.0
            else:
                ego_res[f'raw_density_{z}'] = len(zones[z])
                ego_res[f'raw_speed_{z}'] = np.mean([x[0] for x in zones[z]])
                
                # Occupancy = sum(length + 2) / Area
                occ_sum = sum([x[1] + 2.0 for x in zones[z]])
                if z in ['proceeding', 'following']:
                    occ = min(1.0, occ_sum / 50.0)
                else:
                    # Side zones are 50m long (approx) and rem_lanes wide
                    occ = min(1.0, occ_sum / (50.0 * rem_lanes))
                ego_res[f'relative_occupancy_{z}'] = occ
                
                ego_res[f'relative_speed_{z}'] = ego_res[f'raw_speed_{z}'] / max(1.0, speeds.get(ego_seg, 10.0))
                
        results.append(ego_res)
        
    return pd.DataFrame(results)

def process_single_file(input_file, edges_gdf, args):
    print(f"\n--- Processing {input_file} ---")
    file_start_time = time.time()
    
    df = parse_pneuma_to_long(input_file, test=args.test)
    if df.empty:
        print("No data parsed.")
        return
        
    df = perform_map_matching(df, edges_gdf.crs, 'osm_network.gpkg')
    if df.empty:
        return
        
    # Filter segments with fewer than 10 unique vehicles or in removed_segments
    print(f"Filtering segments (Min 10 vehicles, Exclude blacklist: {len(removed_segments)})...")
    seg_counts = df[df['segment_id'] != ""].groupby('segment_id')['track_id'].nunique()
    
    # Identify segments that meet the volume threshold AND are not in the blacklist
    valid_segments = [s for s in seg_counts.index if s not in removed_segments and seg_counts[s] >= 10]
    
    initial_count = len(df)
    df = df[df['segment_id'].isin(valid_segments)].reset_index(drop=True)
    
    num_removed_segs = len(seg_counts) - len(valid_segments)
    print(f"Removed {initial_count - len(df)} points across {num_removed_segs} segments (low volume or blacklisted).")

    file_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join("processed_data", file_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Export filtered road network
    filtered_network_path = os.path.join(output_dir, 'osm_network.gpkg')
    print(f"Exporting filtered road network to {filtered_network_path}...")
    # Ensure IDs are compared as strings to match valid_segments list
    filtered_edges = edges_gdf[edges_gdf['segment_id'].astype(str).isin(valid_segments)]
    filtered_edges.to_file(filtered_network_path, driver='GPKG')

    df = calculate_signed_distance_and_lanes(df, edges_gdf, output_dir, test=args.test)
    empirical_speeds = calculate_empirical_speeds(df, edges_gdf, output_dir)
    
    # Export matched trajectories for visualization
    matched_trajectories_path = os.path.join(output_dir, "matched_trajectories.csv")
    print(f"Exporting matched trajectories to {matched_trajectories_path}...")
    df.to_csv(matched_trajectories_path, index=False)
    
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
    
    print("Loading topological adjacency...")
    try:
        with open('topological_adjacency.json', 'r') as f:
            adjacency = json.load(f)
    except FileNotFoundError:
        print("Error: topological_adjacency.json not found. Make sure to run the external OSM downloader script.")
        return
        
    # Sampling 15% of all unique Car/Taxi tracks as CAVs
    print("Sampling 15% of unique Car/Taxi tracks as CAVs...")
    df['is_CAV'] = False
    unique_tracks = df[df['type'].isin(['Car', 'Taxi'])]['track_id'].unique()
    n_sample = max(1, int(len(unique_tracks) * 0.15))
    sampled_cavs = np.random.choice(unique_tracks, n_sample, replace=False)
    df.loc[df['track_id'].isin(sampled_cavs), 'is_CAV'] = True
    print(f"Total CAVs selected: {len(sampled_cavs)} out of {len(unique_tracks)} unique Car/Taxi vehicles.")
        
    # Frenet Ego-Centric Extraction
    print("Extracting spatial features via KDTree (using 1-second time buckets)...")
    kdtree_start = time.time()
    
    # Create 1-second time buckets to align vehicles that have jittered timestamps
    df['time_bucket'] = np.round(df['time']).astype(int)
    
    time_groups = []
    for ts, group in df.groupby('time_bucket'):
        time_groups.append((ts, group.copy(), adjacency, empirical_speeds))
        
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(process_frenet_for_timestamp, time_groups)
    pool.close()
    pool.join()
    
    final_df = pd.concat([r for r in results if not r.empty], ignore_index=True)
    print(f"Feature extraction took {time.time() - kdtree_start:.2f} seconds")
    
    if final_df.empty:
        print("No valid CAV features extracted. Exiting.")
        return

    # Deferred Kinematics Calculation (Calculated only for remaining CAVs)
    print("Calculating final kinematics...")
    kin_start = time.time()
    
    final_df['segment_free_flow_speed'] = final_df['segment_id'].map(empirical_speeds).fillna(10.0)
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
        'segment_free_flow_speed', 'relative_ego_speed'
    ]
    
    for z in ['proceeding', 'following', 'leftwards_proceeding', 'leftwards_following', 'rightwards_proceeding', 'rightwards_following']:
        out_cols.extend([f'relative_occupancy_{z}', f'raw_density_{z}', f'relative_speed_{z}', f'raw_speed_{z}'])
    
    final_out_path = os.path.join(output_dir, os.path.basename(input_file).replace('.csv', '_processed.csv'))
    final_df[out_cols].to_csv(final_out_path, index=False)
    
    print(f"\nFinished processing {input_file}! Saved to {final_out_path}")
    print(f"Execution for this file took {time.time() - file_start_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Path to a single pNEUMA CSV file or directory")
    parser.add_argument('--test', action='store_true', help="Test mode (only 15 vehicles)")
    args = parser.parse_args()
    
    overall_start = time.time()
    
    if os.path.isdir(args.input_path):
        files = sorted([os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.csv')])
    else:
        files = [args.input_path]
        
    if not files:
        print("No CSV file(s) found.")
        return
        
    print("Loading OSM network...")
    edges_gdf = gpd.read_file('osm_network.gpkg')
    
    for input_file in files:
        process_single_file(input_file, edges_gdf, args)

    print(f"\nTotal pipeline execution completed in {time.time() - overall_start:.2f} seconds.")

if __name__ == '__main__':
    main()
