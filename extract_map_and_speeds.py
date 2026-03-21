import argparse
import time
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import jenkspy
from shapely.geometry import Point, LineString
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap

sampling_interval = 400  # 1.0Hz interpolation

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
    
    data_list = []
    with open(filepath, 'r') as f:
        # Skip header if present
        first_line = f.readline()
        if "track_id" not in first_line:
            f.seek(0)
            
        count = 0
        for line in f:
            if test and count >= 15:
                break
            
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
            variance = int(sampling_interval * 0.1)  # 10% variance
            current_gap = sampling_interval + np.random.randint(-variance, variance + 1)
            for i in range(0, len(points)-1, 6):
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
                        current_gap = 400 + np.random.randint(-40, 41)
                except ValueError:
                    continue
            
            # Filter duration < 5.0s
            if len(sampled_pts) > 0 and sampled_pts[-1][-1] - sampled_pts[0][-1] >= 5.0:
                for p in sampled_pts:
                    data_list.append([track_id, v_type, traveled_d, avg_speed_ms] + p)
                
            count += 1
            
    df = pd.DataFrame(data_list, columns=[
        'track_id', 'type', 'traveled_d', 'avg_speed', 
        'lat', 'lon', 'speed', 'lon_acc', 'lat_acc', 'time'
    ])
    
    print(f"Parsing took {time.time() - start_time:.2f} seconds. Parsed {len(df)} points.")
    return df

def perform_map_matching(df, map_con, crs, max_dist_start=5, max_dist_end=50, step=5):
    print("Performing iterative map matching...")
    start_time = time.time()
    
    # Project to CRS
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    gdf = gdf.to_crs(crs)
    df['x'] = gdf.geometry.x
    df['y'] = gdf.geometry.y
    
    matched_results = []
    
    # Group by track_id
    grouped = df.groupby('track_id')
    
    for track_id, group in grouped:
        path = group[['x', 'y']].values.tolist()
        
        matcher = None
        matched_states = None
        for max_dist in range(max_dist_start, max_dist_end + 1, step):
            matcher = DistanceMatcher(map_con, max_dist=max_dist, obs_noise=max_dist, obs_noise_ne=max_dist*2)
            states, last_idx = matcher.match(path)
            # Continue trying larger max_dist if we don't have a complete match
            if states and (last_idx == len(path) - 1 or max_dist == max_dist_end):
                matched_states = states
                break
                
        if not matched_states:
            continue
            
        # We need to slice group to match the length of matched_states
        # in case last_idx < len(path) - 1 but we still want to keep the partial match
        group_results = group.iloc[:len(matched_states)].copy()
        
        u_list = []
        v_list = []
        for s in matched_states:
            if isinstance(s, tuple) and len(s) >= 2:
                u_list.append(s[0])
                v_list.append(s[1])
            else:
                u_list.append(s)
                v_list.append(s)
                
        group_results['matched_u'] = u_list
        group_results['matched_v'] = v_list
                
        matched_results.append(group_results)
        
    if not matched_results:
        print("No matches found.")
        return pd.DataFrame()
        
    final_df = pd.concat(matched_results, ignore_index=True)
    print(f"Map matching took {time.time() - start_time:.2f} seconds. Matched {len(final_df)} points.")
    return final_df

def calculate_signed_distance_and_lanes(df, edges_gdf, output_dir):
    print("Calculating lane boundaries...")
    start_time = time.time()
    
    # Map edges to geometries and segment_ids
    edge_map = {}
    lanes_map = {}
    len_map = {}
    for _, row in edges_gdf.iterrows():
        key = (int(row['u']), int(row['v']))
        edge_map[key] = row['geometry']
        lanes_map[key] = int(row['lanes'])
        len_map[key] = float(row['length'])
        
    df['signed_dist'] = np.nan
    df['segment_id'] = ""
    df['num_lanes'] = 0
    df['segment_length'] = 0.0
    
    # Build dictionary from u, v to segment_id
    uv_to_seg = edges_gdf.set_index(['u', 'v'])['segment_id'].to_dict()
    
    for idx, row in df.iterrows():
        u, v = int(row['matched_u']), int(row['matched_v'])
        key = (u, v)
        if key not in edge_map: continue
        
        geom = edge_map[key]
        coords = list(geom.coords)
        
        px, py = row['x'], row['y']
        
        # Simplified: find the closest line segment of the LineString
        best_dist = float('inf')
        signed_d = 0.0
        
        for i in range(len(coords)-1):
            ax, ay = coords[i]
            bx, by = coords[i+1]
            
            p = np.array([px, py])
            a = np.array([ax, ay])
            b = np.array([bx, by])
            
            ab = b - a
            ap = p - a
            
            norm_ab = np.dot(ab, ab)
            if norm_ab == 0: continue
            
            t = max(0, min(1, np.dot(ap, ab) / norm_ab))
            proj = a + t * ab
            dist = np.linalg.norm(p - proj)
            
            if dist < best_dist:
                best_dist = dist
                # Cross product: AB x AP
                cross = ab[0]*ap[1] - ab[1]*ap[0]
                # If cross > 0, AP is to the left of AB. We want negative for left, positive for right.
                # So we take -cross
                sign = -1 if cross > 0 else 1
                signed_d = dist * sign
                
        df.at[idx, 'signed_dist'] = signed_d
        df.at[idx, 'segment_id'] = uv_to_seg.get(key, str(key))
        df.at[idx, 'num_lanes'] = lanes_map.get(key, 2)
        df.at[idx, 'segment_length'] = len_map.get(key, 0.0)
        
    # Process Jenks
    lane_boundaries = {}
    jenks_segments_used = []
    
    grouped = df.groupby('segment_id')
    
    print("Performing Jenks Optimization...")
    for seg_id, group in grouped:
        if seg_id == "": continue
        
        d_vals = group['signed_dist'].values
        max_d = np.percentile(d_vals, 98)
        
        # D = Max_Distance - Vehicle_Distance
        group['D'] = max_d - group['signed_dist']
        group['D'] = group['D'].clip(lower=0.0)
        
        D_vals = group['D'].values
        
        # Get lanes count from OSM (approx)
        u, v = group.iloc[0]['matched_u'], group.iloc[0]['matched_v']
        num_lanes = lanes_map.get((u, v), 2)
        
        if len(D_vals) > 50 and num_lanes > 1:
            sample = np.random.choice(D_vals, min(5000, len(D_vals)), replace=False)
            try:
                breaks = jenkspy.jenks_breaks(sample, n_classes=num_lanes)
                # Enforce widths 2.5m - 4.0m between breaks (heuristic validation)
                valid = True
                for i in range(1, len(breaks)):
                    width = breaks[i] - breaks[i-1]
                    if width < 2.0 or width > 5.0:  # Relaxed for Jenks
                        valid = False
                
                if valid:
                    lane_boundaries[seg_id] = breaks
                    jenks_segments_used.append(seg_id)
                else:
                    lane_boundaries[seg_id] = [i * 3.2 for i in range(num_lanes + 1)]
            except ValueError:
                lane_boundaries[seg_id] = [i * 3.2 for i in range(num_lanes + 1)]
        else:
            # Static fallback
            lane_boundaries[seg_id] = [i * 3.2 for i in range(num_lanes + 1)]
            
        df.loc[group.index, 'D'] = group['D']
        
    # Assign lane index
    df['lane_index'] = 0
    for seg_id, bounds in lane_boundaries.items():
        mask = df['segment_id'] == seg_id
        df.loc[mask, 'lane_index'] = np.digitize(df.loc[mask, 'D'], bins=bounds)
        
    print(f"Lane boundary calculation took {time.time() - start_time:.2f} seconds.")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'lane_boundaries.json'), 'w') as f:
        json.dump(lane_boundaries, f, indent=4)
        
    # Plot 5 debug histograms
    if jenks_segments_used:
        plot_segs = np.random.choice(jenks_segments_used, min(5, len(jenks_segments_used)), replace=False)
        for seg in plot_segs:
            plt.figure(figsize=(8,4))
            d_data = df[df['segment_id'] == seg]['D']
            plt.hist(d_data, bins=50, alpha=0.7)
            for b in lane_boundaries[seg]:
                plt.axvline(b, color='r', linestyle='dashed', linewidth=2)
            plt.title(f'Lane Boundaries for Segment {seg}')
            plt.xlabel('Distance from Right Edge (m)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'histogram_seg_{seg}.png'))
            plt.close()
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Path to a single pNEUMA CSV file or directory")
    parser.add_argument('--test', action='store_true', help="Test mode (only 15 vehicles)")
    args = parser.parse_args()
    
    overall_start = time.time()
    
    if os.path.isdir(args.input_path):
        files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.csv')]
        input_file = files[0] if files else None
    else:
        input_file = args.input_path
        
    if not input_file:
        print("No CSV file found.")
        return
        
    print("Loading OSM network...")
    edges_gdf = gpd.read_file('osm_network.gpkg')
    map_con = get_osm_map(edges_gdf)
    
    df = parse_pneuma_to_long(input_file, test=args.test)
    if df.empty:
        print("No data parsed.")
        return
        
    df = perform_map_matching(df, map_con, edges_gdf.crs)
    if df.empty:
        return
        
    output_dir = "processed_data"
    df = calculate_signed_distance_and_lanes(df, edges_gdf, output_dir)
    calculate_empirical_speeds(df, edges_gdf, output_dir)
    
    # Export matched trajectories for visualization
    matched_trajectories_path = os.path.join(output_dir, "matched_trajectories.csv")
    print(f"Exporting matched trajectories to {matched_trajectories_path}...")
    df.to_csv(matched_trajectories_path, index=False)
    
    print(f"Total initialization completed in {time.time() - overall_start:.2f} seconds.")

if __name__ == '__main__':
    main()
