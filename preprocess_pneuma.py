import multiprocessing
import sys
import time
import os
import csv
import json
import math
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap

chunk_size = 50  # Increased for efficiency
sampling_interval = 1000  # 1.0Hz interpolation

def get_osm_map_and_edges(gpkg_path):
    edges_gdf = gpd.read_file(gpkg_path)
    map_con = InMemMap("osm", use_latlon=False, use_rtree=True, index_edges=True)
    edge_geoms = {}
    for _, row in edges_gdf.iterrows():
        u, v = int(row['u']), int(row['v'])
        coords = list(row['geometry'].coords)
        map_con.add_node(u, (coords[0][0], coords[0][1]))
        map_con.add_node(v, (coords[-1][0], coords[-1][1]))
        map_con.add_edge(u, v)
        edge_geoms[(u, v)] = {
            'segment_id': str(row['segment_id']),
            'length': row['length'],
            'coords': coords,
            'lanes': int(row['lanes']),
            'highway': row['highway']
        }
    return map_con, edges_gdf, edge_geoms

def sample_trajectory(data_points, sampling_interval_ms=1000):
    # points: [(lat, lon, speed, lon_acc, lat_acc, ts), ...]
    pts = []
    last_included_timestamp = None
    variance = int(sampling_interval_ms * 0.1)
    current_gap = sampling_interval_ms + np.random.randint(-variance, variance + 1)
    
    for i in range(0, len(data_points)-1, 6):
        try:
            lat = float(data_points[i].strip())
            lon = float(data_points[i+1].strip())
            speed = float(data_points[i+2].strip()) / 3.6  # km/h to m/s
            lon_acc = float(data_points[i+3].strip())
            lat_acc = float(data_points[i+4].strip())
            ts = float(data_points[i+5].strip())
            
            timestamp_ms = int(ts * 1000)
            
            if last_included_timestamp is None or timestamp_ms - last_included_timestamp >= current_gap:
                pts.append([ts, lat, lon, speed, lon_acc, lat_acc])
                last_included_timestamp = timestamp_ms
                current_gap = sampling_interval_ms + np.random.randint(-variance, variance + 1)
        except:
            pass
            
    if not pts: return []
    
    pts = np.array(pts)
    # Filter duration < 5.0s
    if pts[-1, 0] - pts[0, 0] < 5.0:
        return []
        
    return pts.tolist()

def process_chunk(chunk_data, chunk_index, output_dir, gpkg_path, crs_wkt, lane_bounds_path):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    temp_output_file = os.path.join(output_dir, f'temp_chunk_{chunk_index}.csv')
    
    # Reload globals per worker
    map_con, edges_gdf, edge_geoms = get_osm_map_and_edges(gpkg_path)
    with open(lane_bounds_path, 'r') as f:
        lane_boundaries = json.load(f)
        
    uv_to_seg = {k: v['segment_id'] for k, v in edge_geoms.items()}
    seg_to_uv = {v['segment_id']: k for k, v in edge_geoms.items()}
        
    processed_data = []
    
    for line in chunk_data:
        data = line.strip().split(';')
        if len(data) < 4: continue
        track_id = data[0].strip()
        vehicle_type = data[1].strip()
        
        if vehicle_type == 'Motorcycle': continue
        
        traveled_d = data[2].strip()
        avg_speed_ms = float(data[3].strip()) / 3.6
        
        sampled_pts = sample_trajectory(data[4:], sampling_interval_ms=1000)
        if not sampled_pts: continue
        
        # DataFrame for matching
        df = pd.DataFrame(sampled_pts, columns=['time', 'lat', 'lon', 'speed', 'lon_acc', 'lat_acc'])
        
        # Project to UTM
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        gdf = gdf.to_crs(crs_wkt)
        df['x'] = gdf.geometry.x
        df['y'] = gdf.geometry.y
        
        path = df[['x', 'y']].values.tolist()
        matched_states = None
        for max_dist in range(5, 51, 5):
            matcher = DistanceMatcher(map_con, max_dist=max_dist, obs_noise=max_dist, obs_noise_ne=max_dist*2)
            states, last_idx = matcher.match(path)
            # Continue trying larger max_dist if we don't have a complete match
            if states and (last_idx == len(path) - 1 or max_dist == 50):
                matched_states = states
                break
                
        if not matched_states: continue
        
        # Valid points are those up to the matched states length
        valid_df = df.iloc[:len(matched_states)].copy()
        
        valid_points = []
        for idx, (_, row) in enumerate(valid_df.iterrows()):
            s = matched_states[idx]
            if isinstance(s, tuple) and len(s) >= 2:
                eu, ev = s[0], s[1]
            else:
                eu, ev = s, s
                
            best_edge = (eu, ev)
            if best_edge not in edge_geoms:
                continue
                
            px, py = row['x'], row['y']
            best_dist = float('inf')
            signed_d = 0.0
            t_proj = 0.0
            
            coords = edge_geoms[best_edge]['coords']
            # find closest segment in linestring
            for i in range(len(coords)-1):
                ax, ay = coords[i]
                bx, by = coords[i+1]
                a = np.array([ax, ay])
                b = np.array([bx, by])
                p = np.array([px, py])
                ab = b - a
                ap = p - a
                norm_ab = np.dot(ab, ab)
                if norm_ab == 0: continue
                t = max(0, min(1, np.dot(ap, ab) / norm_ab))
                proj = a + t * ab
                dist = np.linalg.norm(p - proj)
                if dist < best_dist:
                    best_dist = dist
                    cross = ab[0]*ap[1] - ab[1]*ap[0]
                    signed_d = dist * (-1 if cross > 0 else 1)
                    t_proj = np.dot(ap, ab) / np.sqrt(norm_ab) if norm_ab > 0 else 0
                    
            if best_dist > 10.0:
                continue
                
            seg_id = edge_geoms[best_edge]['segment_id']
            
            valid_points.append([
                track_id, vehicle_type, traveled_d, avg_speed_ms,
                row['lat'], row['lon'], row['speed'], row['lon_acc'], row['lat_acc'], row['time'],
                seg_id, edge_geoms[best_edge]['length'], edge_geoms[best_edge]['highway'],
                edge_geoms[best_edge]['lanes'], signed_d, t_proj, row['x'], row['y']
            ])
            
        processed_data.extend(valid_points)
        
    with open(temp_output_file, 'w', newline='') as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for row in processed_data:
            csv_writer.writerow(row)
            
    if chunk_index % 10 == 0: 
        print(f"Chunk {chunk_index} processed in {time.time() - start_time:.2f} seconds")

def process_frenet_for_timestamp(group_data_tuple):
    timestamp, df_group, adjacency, lane_boundaries, speeds = group_data_tuple
    
    if df_group.empty:
        return pd.DataFrame()
        
    # Build KDTree
    coords = df_group[['x', 'y']].values
    tree = KDTree(coords)
    
    cavs = df_group[df_group['is_CAV'] == True]
    if cavs.empty:
        return pd.DataFrame()
        
    results = []
    
    # Pre-calculate vectors for dot product
    # We need segment vectors. Let's approximate from segment length.
    
    for idx, ego in cavs.iterrows():
        ego_idx = df_group.index.get_loc(idx)
        ego_pos = coords[ego_idx]
        ego_seg = ego['segment_id']
        ego_D = ego['D']
        ego_S = ego['t_proj'] # simplistic distance along segment
        
        neighbors_idx = tree.query_ball_point(ego_pos, r=70) # 50m max long + 20m lat
        
        valid_neighbors = []
        for n_idx in neighbors_idx:
            if n_idx == ego_idx: continue
            n_row = df_group.iloc[n_idx]
            n_seg = n_row['segment_id']
            
            # Topological Check
            if n_seg == ego_seg:
                delta_S = n_row['t_proj'] - ego_S
            elif n_seg in adjacency.get(ego_seg, {}).get('successors', []):
                delta_S = (ego['segment_length'] - ego_S) + n_row['t_proj']
            elif n_seg in adjacency.get(ego_seg, {}).get('predecessors', []):
                delta_S = -ego_S - (n_row['segment_length'] - n_row['t_proj'])
            else:
                continue
                
            delta_D = n_row['D'] - ego_D
            valid_neighbors.append({
                'delta_S': delta_S,
                'delta_D': delta_D,
                'speed': n_row['speed'],
                'type': n_row['type']
            })
            
        # Zones
        zones = {'proceeding': [], 'following': [], 'leftwards': [], 'rightwards': []}
        for n in valid_neighbors:
            dS = n['delta_S']
            dD = n['delta_D']
            v_len = 5.0 if n['type'] in ['Car', 'Taxi'] else 12.5 # approx lengths
            
            if 0 < dS <= 50 and -1.6 <= dD <= 1.6:
                zones['proceeding'].append((n['speed'], v_len))
            elif -50 <= dS < 0 and -1.6 <= dD <= 1.6:
                zones['following'].append((n['speed'], v_len))
            elif -50 <= dS <= 50 and dD > 1.6:
                zones['leftwards'].append((n['speed'], v_len))
            elif -50 <= dS <= 50 and dD < -1.6:
                zones['rightwards'].append((n['speed'], v_len))
                
        ego_res = ego.to_dict()
        rem_lanes = max(1, ego['num_lanes'] - 1)
        
        for z in ['proceeding', 'following', 'leftwards', 'rightwards']:
            if not zones[z]:
                ego_res[f'raw_density_{z}'] = 0
                ego_res[f'raw_speed_{z}'] = speeds.get(ego_seg, 10.0) # Impute free flow
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
                    occ = min(1.0, occ_sum / (100.0 * rem_lanes))
                ego_res[f'relative_occupancy_{z}'] = occ
                
                ego_res[f'relative_speed_{z}'] = ego_res[f'raw_speed_{z}'] / speeds.get(ego_seg, 10.0)
                
        results.append(ego_res)
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="Input CSV or directory")
    parser.add_argument('--test', action='store_true', help="Test mode (only 15 vehicles)")
    args = parser.parse_args()
    
    input_file = args.input_path
    if os.path.isdir(args.input_path):
        input_file = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.endswith('.csv')][0]
        
    output_dir = "temp_processing"
    
    print("Loading initial networks and parameters...")
    init_start = time.time()
    gpkg_path = 'osm_network.gpkg'
    edges_gdf = gpd.read_file(gpkg_path)
    crs_wkt = edges_gdf.crs.to_wkt()
    
    with open('topological_adjacency.json', 'r') as f:
        adjacency = json.load(f)
    with open('processed_data/lane_boundaries.json', 'r') as f:
        lane_boundaries = json.load(f)
    with open('processed_data/empirical_free_flow_speeds.json', 'r') as f:
        speeds = json.load(f)
        
    print(f"Initialization took {time.time() - init_start:.2f} seconds")
    
    # 1. Chunked Map Matching
    print("Starting map matching and parsing...")
    parse_start = time.time()
    
    with open(input_file, 'r') as file:
        lines = file.readlines()
        if args.test: lines = lines[:16]
        chunks = [lines[i:i + chunk_size] for i in range(1, len(lines), chunk_size)]
        
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for idx, chunk in enumerate(chunks):
        pool.apply_async(process_chunk, args=(chunk, idx, output_dir, gpkg_path, crs_wkt, 'processed_data/lane_boundaries.json'))
    pool.close()
    pool.join()
    
    print(f"Map matching took {time.time() - parse_start:.2f} seconds")
    
    # 2. Merge Chunks and Compute Global D
    print("Merging chunks and assigning lanes...")
    merge_start = time.time()
    all_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('temp_chunk_')]
    df_list = []
    columns = ['track_id', 'type', 'traveled_distance', 'avg_speed', 'lat', 'lon', 'speed', 'long_acc', 'lat_acc', 'time',
               'segment_id', 'segment_length', 'segment_type', 'num_lanes', 'signed_dist', 't_proj', 'x', 'y']
               
    for f in all_files:
        df_list.append(pd.read_csv(f, names=columns))
        os.remove(f)
        
    if not df_list:
        print("No valid data processed.")
        return
        
    df = pd.concat(df_list, ignore_index=True)
    
    # Global Anchor
    df['D'] = 0.0
    for seg_id, group in df.groupby('segment_id'):
        max_d = np.percentile(group['signed_dist'], 98)
        df.loc[group.index, 'D'] = (max_d - group['signed_dist']).clip(lower=0.0)
        
    # O(1) Lane Assignment
    df['lane_index'] = 0
    for seg_id, bounds in lane_boundaries.items():
        mask = df['segment_id'] == seg_id
        df.loc[mask, 'lane_index'] = np.digitize(df.loc[mask, 'D'], bins=bounds)
        
    print(f"Merge and lane assignment took {time.time() - merge_start:.2f} seconds")
    
    # 3. Stratified Sampling
    print("Performing stratified sampling...")
    df['time_bin'] = (df['time'] // 300) * 300
    df['is_CAV'] = False
    
    for _, group in df[df['type'].isin(['Car', 'Taxi'])].groupby('time_bin'):
        unique_tracks = group['track_id'].unique()
        n_sample = max(1, int(len(unique_tracks) * 0.15))
        sampled = np.random.choice(unique_tracks, n_sample, replace=False)
        df.loc[df['track_id'].isin(sampled), 'is_CAV'] = True
        
    # 4. Frenet Ego-Centric Extraction
    print("Extracting spatial features via KDTree...")
    kdtree_start = time.time()
    
    time_groups = []
    for ts, group in df.groupby('time'):
        time_groups.append((ts, group.copy(), adjacency, lane_boundaries, speeds))
        
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(process_frenet_for_timestamp, time_groups)
    pool.close()
    pool.join()
    
    final_df = pd.concat([r for r in results if not r.empty], ignore_index=True)
    print(f"Feature extraction took {time.time() - kdtree_start:.2f} seconds")
    
    # 5. Deferred Kinematics
    print("Calculating final kinematics...")
    kin_start = time.time()
    
    final_df['segment_free_flow_speed'] = final_df['segment_id'].map(speeds).fillna(10.0)
    final_df['relative_ego_speed'] = final_df['speed'] / final_df['segment_free_flow_speed']
    final_df['proportionate_distance_travelled'] = final_df['t_proj'] / final_df['segment_length']
    
    final_df = final_df.sort_values(['track_id', 'time'])
    
    final_df['change_in_euclidean_distance'] = final_df.groupby('track_id').apply(
        lambda x: np.sqrt((x['x'].diff()**2) + (x['y'].diff()**2))
    ).reset_index(level=0, drop=True)
    
    final_df['relative_time_gap'] = final_df.groupby('track_id')['time'].diff()
    
    final_df['relative_kinematic_ratio'] = final_df['change_in_euclidean_distance'] / (final_df['segment_free_flow_speed'] * final_df['relative_time_gap'])
    final_df['relative_kinematic_ratio'] = final_df['relative_kinematic_ratio'].clip(upper=1.0)
    
    print(f"Kinematics took {time.time() - kin_start:.2f} seconds")
    
    # Export
    out_cols = [
        'track_id', 'type', 'traveled_distance', 'avg_speed', 'lat', 'lon', 'speed', 'long_acc', 'lat_acc', 'time',
        'segment_id', 'segment_length', 'segment_type', 'num_lanes', 'lane_index', 'proportionate_distance_travelled',
        'change_in_euclidean_distance', 'relative_time_gap', 'relative_kinematic_ratio',
        'relative_occupancy_proceeding', 'relative_occupancy_following', 'relative_occupancy_leftwards', 'relative_occupancy_rightwards',
        'raw_density_proceeding', 'raw_density_following', 'raw_density_leftwards', 'raw_density_rightwards',
        'segment_free_flow_speed', 'relative_ego_speed',
        'relative_speed_proceeding', 'relative_speed_following', 'relative_speed_leftwards', 'relative_speed_rightwards',
        'raw_speed_proceeding', 'raw_speed_following', 'raw_speed_leftwards', 'raw_speed_rightwards'
    ]
    
    final_out_path = os.path.basename(input_file).replace('.csv', '_processed.csv')
    final_df[out_cols].to_csv(final_out_path, index=False)
    print(f"Pipeline finished! Saved to {final_out_path}")

if __name__ == '__main__':
    main()
