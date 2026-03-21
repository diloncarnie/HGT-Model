import multiprocessing
import sys
import time
import os
import csv
import argparse
import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
import jenkspy
import json

chunk_size = 50 

def parse_chunk(chunk_data, chunk_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    temp_output_file = os.path.join(output_dir, f'temp_chunk_{chunk_index}.csv')
    
    processed_data = []
    for line in chunk_data:
        data = line.strip().split(';')
        if len(data) < 4:
            continue
        track_id = data[0].strip()
        vehicle_type = data[1].strip()
        traveled_d = data[2].strip()
        try:
            avg_speed_kmh = float(data[3].strip())
            avg_speed_ms = avg_speed_kmh / 3.6
        except ValueError:
            avg_speed_ms = 0.0
            
        data_points = data[4:]
        
        for i in range(0, len(data_points)-1, 6):
            try:
                lat = data_points[i].strip()
                lon = data_points[i+1].strip()
                speed_kmh = float(data_points[i+2].strip())
                speed_ms = speed_kmh / 3.6
                long_acc = data_points[i+3].strip()
                lat_acc = data_points[i+4].strip()
                timestamp = data_points[i+5].strip()
                
                processed_data.append([
                    track_id, vehicle_type, traveled_d, avg_speed_ms, 
                    lat, lon, speed_ms, 
                    long_acc, lat_acc, timestamp
                ])
            except (ValueError, IndexError):
                continue

    with open(temp_output_file, 'w', newline='') as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for row in processed_data:
            csv_writer.writerow(row)

def parse_file_to_long(input_file, output_dir, is_test=False):
    parsed_dir = os.path.join(output_dir, "parsed_chunks_" + os.path.basename(input_file))
    os.makedirs(parsed_dir, exist_ok=True)
    
    with open(input_file, 'r') as file:
        lines = file.readlines()
        if is_test:
            lines = lines[:15] # Take header + a few lines
        chunks = [lines[i:i + chunk_size] for i in range(1, len(lines), chunk_size)]
        
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for chunk_index, chunk_data in enumerate(chunks):
        pool.apply_async(parse_chunk, args=(chunk_data, chunk_index, parsed_dir))

    pool.close()
    pool.join()
    
    output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.csv', '_long.csv'))
    
    def sort_key(filename):
        parts = filename.split('_')
        if len(parts) >= 3 and parts[-1].split('.')[0].isdigit():
            return int(parts[-1].split('.')[0])
        return float('inf')

    with open(output_file, 'w', newline='') as f_out:
        f_out.write('track_id,type,traveled_d,avg_speed,lat,lon,speed,long_acc,lat_acc,time\n') 
        
        for filename in sorted(os.listdir(parsed_dir), key=sort_key):
            if filename.startswith('temp_chunk_'):
                filepath = os.path.join(parsed_dir, filename)
                with open(filepath, 'r') as f_in:
                    f_out.writelines(f_in.readlines()) 
                os.remove(filepath)
    
    os.rmdir(parsed_dir)
    return output_file

def get_osm_network():
    print("Downloading OSM network for Athens...")
    try:
        graph = ox.graph_from_place('Athens, Greece', network_type='drive')
    except Exception as e:
        # print(f"Failed to download Athens using place name, using bbox... Error: {e}")
        # Bounding box roughly covering Athens center
        graph = ox.graph_from_bbox(bbox=(23.70, 37.95, 23.76, 38.00), network_type='drive')

    valid_highway_types = ['primary', 'secondary', 'tertiary', 'trunk']
    edges_to_keep = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        hw = data.get('highway')
        if isinstance(hw, list):
            hw = hw[0]
        if hw in valid_highway_types:
            edges_to_keep.append((u, v, k))
            
    filtered_graph = graph.edge_subgraph(edges_to_keep).copy()
    graph_utm = ox.project_graph(filtered_graph)
    
    nodes, edges = ox.graph_to_gdfs(graph_utm)
    edges = edges.reset_index()
    edges['segment_id'] = edges.index.astype(str)
    
    if 'length' not in edges.columns:
        edges['length'] = edges.geometry.length
    if 'lanes' not in edges.columns:
        edges['lanes'] = 2
    else:
        edges['lanes'] = pd.to_numeric(edges['lanes'], errors='coerce').fillna(2)
        
    def get_first_hw(x):
        return x[0] if isinstance(x, list) else x
    edges['highway'] = edges['highway'].apply(get_first_hw)
    
    return graph_utm, edges

def process_pass1_file(file, edges):
    print(f"  Processing {file} for Pass 1...")
    local_speeds = []
    df = pd.read_csv(file)
    df = df[df['type'] == 'Car'].copy()
    if df.empty:
        return local_speeds
        
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326")
    gdf = gdf.to_crs(edges.crs)
    
    # Merge with edge nearest
    matched = gpd.sjoin_nearest(gdf, edges[['segment_id', 'geometry']], how='left', distance_col='dist')
    matched = matched[~matched.index.duplicated(keep='first')]
    matched = matched.merge(edges[['segment_id', 'geometry']], on='segment_id', suffixes=('', '_edge'))
    
    def safe_project(row):
        try:
            return row['geometry_edge'].project(row['geometry'])
        except:
            return 0.0
            
    matched['raw_offset'] = matched.apply(safe_project, axis=1)
    
    for (seg_id, t_id), group in matched.groupby(['segment_id', 'track_id']):
        if len(group) < 2:
            continue
        delta_offset = abs(group['raw_offset'].max() - group['raw_offset'].min())
        delta_time = group['time'].max() - group['time'].min()
        if delta_time > 0 and delta_offset > 0:
            speed = delta_offset / delta_time
            local_speeds.append({'segment_id': seg_id, 'speed': speed})
            
    return local_speeds

def pass1_global_speeds(long_files, edges):
    print("Starting Pass 1: Global Free-Flow Speed Derivation...")
    all_speeds = []
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_pass1_file, [(f, edges) for f in long_files])
        
    for res in results:
        all_speeds.extend(res)
                
    if not all_speeds:
        print("Warning: No speeds calculated in Pass 1. Defaulting to 15 m/s.")
        return {seg: 15.0 for seg in edges['segment_id'].unique()}
        
    speed_df = pd.DataFrame(all_speeds)
    global_speeds = speed_df.groupby('segment_id')['speed'].quantile(0.95).to_dict()
    
    global_avg = np.mean(list(global_speeds.values()))
    for seg_id in edges['segment_id'].unique():
        if seg_id not in global_speeds:
            global_speeds[seg_id] = global_avg
            
    return global_speeds

def process_timestamp_chunk(t_group_data, adjacency, global_speeds):
    t_val, t_group = t_group_data
    results = []
    coords = np.array([(p.x, p.y) for p in t_group['geometry']])
    if len(coords) == 0:
        return results
    tree = KDTree(coords)
    t_group_reset = t_group.reset_index(drop=True)
    cavs = t_group_reset[t_group_reset['is_CAV']]
    for idx, ego in cavs.iterrows():
        ego_coord = (ego['geometry'].x, ego['geometry'].y)
        neighbor_indices = tree.query_ball_point(ego_coord, 100.0)
        neighbors = t_group_reset.iloc[neighbor_indices]
        neighbors = neighbors[neighbors['track_id'] != ego['track_id']]
        
        ego_seg = str(ego['segment_id'])
        adj = adjacency.get(ego_seg, {'successors': [], 'predecessors': []})
        valid_segs = [ego_seg] + adj['successors'] + adj['predecessors']
        neighbors = neighbors[neighbors['segment_id'].astype(str).isin(valid_segs)]
        
        dot_product = np.cos(ego['heading']) * np.cos(neighbors['heading']) + np.sin(ego['heading']) * np.sin(neighbors['heading'])
        neighbors = neighbors[dot_product > 0]
        
        zonal_stats = {'proceeding': [], 'following': [], 'leftwards': [], 'rightwards': []}
        for _, nb in neighbors.iterrows():
            delta_d = nb['distance_from_centerline'] - ego['distance_from_centerline']
            
            nb_seg = str(nb['segment_id'])
            if nb_seg == ego_seg:
                delta_s = nb['raw_offset'] - ego['raw_offset']
            elif nb_seg in adj['successors']:
                delta_s = (ego['segment_length'] - ego['raw_offset']) + nb['raw_offset']
            elif nb_seg in adj['predecessors']:
                delta_s = -(ego['raw_offset'] + (nb['segment_length'] - nb['raw_offset']))
            else:
                continue
                
            # Frenet Zonal Bounding Boxes
            if 0 < delta_s <= 50 and abs(delta_d) <= 1.8:
                zonal_stats['proceeding'].append(nb)
            elif -50 <= delta_s < 0 and abs(delta_d) <= 1.8:
                zonal_stats['following'].append(nb)
            elif -50 <= delta_s <= 50 and delta_d > 1.8:
                zonal_stats['leftwards'].append(nb)
            elif -50 <= delta_s <= 50 and delta_d < -1.8:
                zonal_stats['rightwards'].append(nb)
                
        zones = {}
        for z, nbs in zonal_stats.items():
            if len(nbs) == 0:
                zones[f'raw_density_{z}'] = 0
                zones[f'raw_speed_{z}'] = np.nan
                zones[f'relative_occupancy_{z}'] = 0.0
            else:
                nb_df = pd.DataFrame(nbs)
                zones[f'raw_density_{z}'] = len(nb_df)
                zones[f'raw_speed_{z}'] = nb_df['speed'].mean()
                total_length = len(nb_df) * 7.0
                if z in ['proceeding', 'following']:
                    occ = min(1.0, total_length / 50.0)
                else:
                    remaining_lanes = max(1, int(ego['num_lanes']) - 1)
                    occ = min(1.0, total_length / (100.0 * remaining_lanes))
                zones[f'relative_occupancy_{z}'] = occ
        row_data = ego.to_dict()
        row_data.update(zones)
        results.append(row_data)
    return results

def process_pass2(file, edges, adjacency, global_speeds):
    print(f"Starting Pass 2 on {file}...")
    df = pd.read_csv(file)
    
    df = df[df['type'] != 'Motorcycle'].copy()
    
    durations = df.groupby('track_id')['time'].agg(['min', 'max'])
    valid_tracks = durations[(durations['max'] - durations['min']) >= 5.0].index
    df = df[df['track_id'].isin(valid_tracks)].copy()
    
    if df.empty:
        print(f"No valid tracks found in {file} after duration filter.")
        return
        
    interpolated_dfs = []
    for track_id, group in df.groupby('track_id'):
        group = group.sort_values('time').drop_duplicates('time')
        if len(group) < 2:
            continue
        t_min, t_max = group['time'].min(), group['time'].max()
        t_grid = np.arange(np.ceil(t_min), np.floor(t_max) + 1, 1.0)
        if len(t_grid) == 0:
            continue
            
        interp_func_lat = interp1d(group['time'], group['lat'], kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_func_lon = interp1d(group['time'], group['lon'], kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_func_speed = interp1d(group['time'], group['speed'], kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_func_lon_acc = interp1d(group['time'], group['long_acc'], kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_func_lat_acc = interp1d(group['time'], group['lat_acc'], kind='linear', bounds_error=False, fill_value='extrapolate')
        
        v_type = group['type'].iloc[0]
        v_td = group['traveled_d'].iloc[0]
        v_as = group['avg_speed'].iloc[0]
        
        interp_df = pd.DataFrame({
            'track_id': track_id,
            'type': v_type,
            'traveled_d': v_td,
            'avg_speed': v_as,
            'lat': interp_func_lat(t_grid),
            'lon': interp_func_lon(t_grid),
            'speed': interp_func_speed(t_grid),
            'long_acc': interp_func_lon_acc(t_grid),
            'lat_acc': interp_func_lat_acc(t_grid),
            'time': t_grid
        })
        interpolated_dfs.append(interp_df)
        
    if not interpolated_dfs:
        return
    df = pd.concat(interpolated_dfs, ignore_index=True)
    
    df['time'] = df['time'] + np.random.uniform(-0.2, 0.2, size=len(df))
    df['time'] = df['time'].clip(lower=0.0)
    df = df.sort_values(['track_id', 'time'])
    df['relative_time_gap'] = df.groupby('track_id')['time'].diff().fillna(1.0)
    df['relative_time_gap'] = df['relative_time_gap'].clip(lower=0.15)
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326")
    gdf = gdf.to_crs(edges.crs)
    
    matched = gpd.sjoin_nearest(gdf, edges[['segment_id', 'length', 'highway', 'lanes', 'geometry']], how='left', distance_col='dist')
    matched = matched[~matched.index.duplicated(keep='first')]
    matched = matched.merge(edges[['segment_id', 'geometry']], on='segment_id', suffixes=('', '_edge'))
    
    def signed_cross_dist(row):
        P = row['geometry']
        edge = row['geometry_edge']
        min_dist = float('inf')
        closest_cross = 0
        coords = list(edge.coords)
        for i in range(len(coords)-1):
            A, B = Point(coords[i]), Point(coords[i+1])
            seg = LineString([A, B])
            d = P.distance(seg)
            if d < min_dist:
                min_dist = d
                cross = (B.x - A.x)*(P.y - A.y) - (B.y - A.y)*(P.x - A.x)
                closest_cross = cross
        return (-1 if closest_cross > 0 else 1) * min_dist

    matched['distance_from_centerline'] = matched.apply(signed_cross_dist, axis=1)
    
    def raw_offset_calc(row):
        try:
            return row['geometry_edge'].project(row['geometry'])
        except:
            return 0.0
    matched['raw_offset'] = matched.apply(raw_offset_calc, axis=1)
    
    matched['segment_length'] = matched['length']
    matched['segment_type'] = matched['highway']
    matched['num_lanes'] = matched['lanes']
    
    # Jenks-based Lane Detection
    print("  Applying Natural Breaks lane detection...")
    def assign_lanes_jenks(seg_group):
        sid = seg_group.name
        car_taxi = seg_group[seg_group['type'].isin(['Car', 'Taxi'])]
        d_vals = car_taxi['distance_from_centerline'].dropna().values
        n_lanes = max(1, int(seg_group['num_lanes'].iloc[0]))
        if len(d_vals) < n_lanes * 2:
            seg_group['lane_index'] = 0
            return seg_group
        if len(d_vals) > 5000:
            d_vals = np.random.choice(d_vals, 5000, replace=False)
        try:
            breaks = jenkspy.jenks_breaks(d_vals, n_classes=n_lanes)
            seg_group['lane_index'] = np.digitize(seg_group['distance_from_centerline'], breaks) - 1
            seg_group['lane_index'] = seg_group['lane_index'].clip(lower=0, upper=n_lanes-1)
        except:
            seg_group['lane_index'] = 0
        seg_group['segment_id'] = sid
        return seg_group
        
    matched = matched.groupby('segment_id', group_keys=False).apply(assign_lanes_jenks)
    
    matched['proportionate_distance_travelled'] = matched['raw_offset'] / matched['segment_length'].replace(0, 1)
    
    matched['prev_geom'] = matched.groupby('track_id')['geometry'].shift(1)
    matched['change_in_euclidean_distance'] = matched.apply(lambda r: r['geometry'].distance(r['prev_geom']) if pd.notnull(r['prev_geom']) else 0.0, axis=1)
    matched.drop(columns=['prev_geom'], inplace=True)
    
    matched['prev_x'] = matched.groupby('track_id')['geometry'].shift(1).apply(lambda p: p.x if pd.notnull(p) else np.nan)
    matched['prev_y'] = matched.groupby('track_id')['geometry'].shift(1).apply(lambda p: p.y if pd.notnull(p) else np.nan)
    matched['dx'] = matched['geometry'].x - matched['prev_x']
    matched['dy'] = matched['geometry'].y - matched['prev_y']
    matched['dx'] = matched.groupby('track_id')['dx'].bfill().fillna(0.001)
    matched['dy'] = matched.groupby('track_id')['dy'].bfill().fillna(0.001)
    matched['heading'] = np.arctan2(matched['dy'], matched['dx'])
    
    cars = matched[matched['type'].isin(['Car', 'Taxi'])]
    entry_times = cars.groupby('track_id')['time'].min()

    if entry_times.empty:
        print("No cars/taxis found for CAV sampling.")
        return

    bins = np.arange(entry_times.min(), entry_times.max() + 300, 300)
    if len(bins) == 1:
        bins = np.append(bins, bins[0] + 300)
    entry_bins = pd.cut(entry_times, bins=bins, include_lowest=True)

    cav_ids = []
    for bin_val, group in entry_times.groupby(entry_bins, observed=False):
        n_sample = max(1, int(len(group) * 0.15))
        if len(group) > 0:
            sampled = np.random.choice(group.index, size=n_sample, replace=False)
            cav_ids.extend(sampled)
            
    matched['is_CAV'] = matched['track_id'].isin(cav_ids)
    matched['rounded_time'] = matched['time'].round()
    
    results = []
    time_groups = [(t_val, t_group) for t_val, t_group in matched.groupby('rounded_time')]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        chunk_results = pool.starmap(process_timestamp_chunk, [(g, adjacency, global_speeds) for g in time_groups])
        
    for res_list in chunk_results:
        results.extend(res_list)
            
    final_df = pd.DataFrame(results)
    if final_df.empty:
        print(f"No CAV data in {file}.")
        return
        
    final_df['segment_free_flow_speed'] = final_df['segment_id'].map(global_speeds).fillna(15.0)
    
    final_df['relative_kinematic_ratio'] = final_df.apply(
        lambda r: min(1.0, r['change_in_euclidean_distance'] / (max(0.1, r['segment_free_flow_speed']) * r['relative_time_gap'])),
        axis=1
    )
    
    final_df['relative_ego_speed'] = final_df['speed'] / final_df['segment_free_flow_speed'].replace(0, 0.15)
    
    for z in ['proceeding', 'following', 'leftwards', 'rightwards']:
        raw_speed_col = f'raw_speed_{z}'
        rel_speed_col = f'relative_speed_{z}'
        final_df[rel_speed_col] = final_df[raw_speed_col] / final_df['segment_free_flow_speed'].replace(0, 0.15)
        final_df[rel_speed_col] = final_df[rel_speed_col].fillna(1.0)
        final_df[raw_speed_col] = final_df[raw_speed_col].fillna(0.0)
        
    final_df.rename(columns={'traveled_d': 'traveled_distance'}, inplace=True)

    final_cols = [
        'track_id', 'type', 'traveled_distance', 'avg_speed', 'lat', 'lon', 'speed', 'long_acc', 'lat_acc', 'time',
        'segment_id', 'segment_length', 'segment_type', 'num_lanes', 'lane_index', 
        'proportionate_distance_travelled', 'change_in_euclidean_distance', 'relative_time_gap', 
        'relative_kinematic_ratio', 
        'relative_occupancy_proceeding', 'relative_occupancy_following', 'relative_occupancy_leftwards', 'relative_occupancy_rightwards', 
        'raw_density_proceeding', 'raw_density_following', 'raw_density_leftwards', 'raw_density_rightwards', 
        'segment_free_flow_speed', 'relative_ego_speed', 
        'relative_speed_proceeding', 'relative_speed_following', 'relative_speed_leftwards', 'relative_speed_rightwards', 
        'raw_speed_proceeding', 'raw_speed_following', 'raw_speed_leftwards', 'raw_speed_rightwards'
    ]
    
    for c in final_cols:
        if c not in final_df.columns:
            final_df[c] = np.nan
            
    final_df = final_df[final_cols]
    final_df = final_df.sort_values(['time', 'track_id'])
    
    out_file = file.replace('_long.csv', '_processed.csv')
    final_df.to_csv(out_file, index=False)
    print(f"Exported {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Process pNEUMA dataset")
    parser.add_argument('input', help="Input CSV file or directory")
    parser.add_argument('--out_dir', default='.', help="Output directory for processed files")
    parser.add_argument('--test', action='store_true', help="Run in test mode (processes only a few lines)")
    args = parser.parse_args()
    
    input_path = args.input
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    files_to_process = []
    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            if f.endswith('.csv') and not f.endswith('_long.csv') and not f.endswith('_processed.csv'):
                files_to_process.append(os.path.join(input_path, f))
    else:
        files_to_process.append(input_path)
        
    if not files_to_process:
        print("No CSV files found.")
        sys.exit(1)
        
    print(f"Found {len(files_to_process)} files to process.")
    
    long_files = []
    for file in files_to_process:
        print(f"Parsing {file} to long format...")
        long_file = parse_file_to_long(file, out_dir, is_test=args.test)
        long_files.append(long_file)
        
    graph_utm, edges = get_osm_network()
    
    # Build topological adjacency map
    print("Building topological adjacency map...")
    adjacency = {}
    for _, row in edges.iterrows():
        sid = str(row['segment_id'])
        u, v = row['u'], row['v']
        successors = edges[edges['u'] == v]['segment_id'].astype(str).tolist()
        predecessors = edges[edges['v'] == u]['segment_id'].astype(str).tolist()
        adjacency[sid] = {'successors': successors, 'predecessors': predecessors}
    
    # Save the deterministic OSM network for visualization
    network_path = os.path.join(out_dir, "osm_network.gpkg")
    print(f"Saving frozen OSM network to {network_path}...")
    edges_to_save = edges.copy()
    for col in edges_to_save.columns:
        if edges_to_save[col].apply(lambda x: isinstance(x, list)).any():
            edges_to_save[col] = edges_to_save[col].astype(str)
    edges_to_save.to_file(network_path, driver="GPKG")
    
    start_p1 = time.time()
    global_speeds = pass1_global_speeds(long_files, edges)
    end_p1 = time.time()
    print(f"Pass 1 completed in {end_p1 - start_p1:.2f} seconds.")
    
    start_p2 = time.time()
    for file in long_files:
        process_pass2(file, edges, adjacency, global_speeds)
    end_p2 = time.time()
    print(f"Pass 2 completed in {end_p2 - start_p2:.2f} seconds.")
        
if __name__ == '__main__':
    main()
