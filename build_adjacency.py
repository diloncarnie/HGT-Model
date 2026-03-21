import geopandas as gpd
import pandas as pd
import osmnx as ox
import math
import json
import argparse
import os
import numpy as np

def get_osm_network():
    print("Downloading OSM network for Athens...")
    try:
        graph = ox.graph_from_place('Athens, Greece', network_type='drive')
    except Exception as e:
        # Bounding box covering Athens center
        graph = ox.graph_from_bbox(bbox=(23.70, 37.95, 23.76, 38.00), network_type='drive')

    valid_highway_types = ['primary', 'secondary', 'tertiary', 'trunk', 'residential', 'unclassified']
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
    
    return edges

def get_smoothed_heading(geom, reverse=False, look_dist=10.0):
    """
    Calculates a stable heading by looking back/forward 'look_dist' meters 
    along the vertices from the junction node.
    """
    coords = list(geom.coords)
    if len(coords) < 2:
        return 0.0
    
    if not reverse:
        # EXIT HEADING: Look back from the end point (coords[-1])
        ref_pt = coords[-1]
        target_pt = coords[-2]
        accum_dist = 0.0
        for i in range(len(coords) - 1, 0, -1):
            p_end = coords[i]
            p_start = coords[i-1]
            d = math.sqrt((p_end[0]-p_start[0])**2 + (p_end[1]-p_start[1])**2)
            if accum_dist + d >= look_dist:
                target_pt = p_start
                break
            accum_dist += d
            target_pt = p_start
        p1, p2 = target_pt, ref_pt
    else:
        # ENTRY HEADING: Look ahead from the start point (coords[0])
        ref_pt = coords[0]
        target_pt = coords[1]
        accum_dist = 0.0
        for i in range(0, len(coords) - 1):
            p_start = coords[i]
            p_end = coords[i+1]
            d = math.sqrt((p_end[0]-p_start[0])**2 + (p_end[1]-p_start[1])**2)
            if accum_dist + d >= look_dist:
                target_pt = p_end
                break
            accum_dist += d
            target_pt = p_end
        p1, p2 = ref_pt, target_pt
        
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def build_topological_adjacency(edges_gdf):
    print("Building topological lookups...")
    # Grouping by node IDs
    edges_by_u = edges_gdf.groupby('u')['segment_id'].apply(list).to_dict()
    edges_by_v = edges_gdf.groupby('v')['segment_id'].apply(list).to_dict()
    
    adjacency = {}
    
    print("Pre-calculating smoothed headings (10m window)...")
    segment_data = {}
    for _, row in edges_gdf.iterrows():
        sid = str(row['segment_id'])
        geom = row['geometry']
        segment_data[sid] = {
            'u': row['u'],
            'v': row['v'],
            'osmid': str(row['osmid']),
            'exit_heading': get_smoothed_heading(geom, reverse=False),
            'entry_heading': get_smoothed_heading(geom, reverse=True)
        }

    print("Analyzing topological connections...")
    for ego_id, data in segment_data.items():
        u_node, v_node = data['u'], data['v']
        ego_osmid = data['osmid']
        
        valid_successors = []
        valid_predecessors = []
        
        # 1. SUCCESSORS: segments starting where we end (v_node)
        potential_successors = edges_by_u.get(v_node, [])
        for succ_id in potential_successors:
            succ_id_str = str(succ_id)
            if succ_id_str == ego_id: continue
            
            # Heading Check
            succ_heading = segment_data[succ_id_str]['entry_heading']
            delta_theta = (succ_heading - data['exit_heading'] + 180) % 360 - 180
            
            # Same OSM Way Check (Priority)
            is_same_way = segment_data[succ_id_str]['osmid'] == ego_osmid
            threshold = 60.0 if is_same_way else 45.0
            
            if abs(delta_theta) <= threshold:
                valid_successors.append(succ_id_str)
                
        # 2. PREDECESSORS: segments ending where we start (u_node)
        potential_predecessors = edges_by_v.get(u_node, [])
        for pred_id in potential_predecessors:
            pred_id_str = str(pred_id)
            if pred_id_str == ego_id: continue
            
            pred_heading = segment_data[pred_id_str]['exit_heading']
            delta_theta = (data['entry_heading'] - pred_heading + 180) % 360 - 180
            
            is_same_way = segment_data[pred_id_str]['osmid'] == ego_osmid
            threshold = 60.0 if is_same_way else 45.0
            
            if abs(delta_theta) <= threshold:
                valid_predecessors.append(pred_id_str)
                
        adjacency[ego_id] = {
            'successors': valid_successors,
            'predecessors': valid_predecessors
        }
        
    return adjacency

def main():
    parser = argparse.ArgumentParser(description="Build robust topological adjacency. Always downloads the latest network.")
    parser.add_argument('--output', default='topological_adjacency.json', help="Output JSON filename.")
    args = parser.parse_args()
    
    # Always download the network as requested
    edges = get_osm_network()
    network_path = "osm_network.gpkg"
    print(f"Saving frozen OSM network to {network_path}...")
    
    # GPKG cannot handle lists well. Convert any list-like osmid or highway tags to simple strings for saving.
    edges_to_save = edges.copy()
    for col in edges_to_save.columns:
        if edges_to_save[col].apply(lambda x: isinstance(x, list)).any():
            edges_to_save[col] = edges_to_save[col].astype(str)
            
    edges_to_save.to_file(network_path, driver="GPKG")
    
    adjacency = build_topological_adjacency(edges)
    
    print(f"Exporting adjacency to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(adjacency, f, indent=4)
    print("Done.")

if __name__ == '__main__':
    main()