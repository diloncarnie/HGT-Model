import geopandas as gpd
import pandas as pd
import osmnx as ox
import math
import json
import argparse
import os
import numpy as np
import requests
import ast
from shapely.geometry import LineString
import time
import logging

# Set up the global Python logger
logger = logging.getLogger("build_adjacency")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("build-adjacency.log", mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

def get_osm_network():
    # Instruct OSMnx to retain advanced routing and road rule tags
    ox.settings.useful_tags_way += ['turn:lanes', 'turn', 'junction', 'access']
    
    logger.info("Downloading OSM network for Athens...")
    try:
            graph = ox.graph_from_place('Athens, Greece', network_type='all', simplify=False)
    except Exception as e:
        # Bounding box covering Athens center
            graph = ox.graph_from_bbox(bbox=(23.70, 37.95, 23.76, 38.00), network_type='all', simplify=False)

    valid_highway_types = ['primary', 'secondary', 'tertiary', 'trunk', 'residential', 'unclassified', 'service', 'living_street', 'road', 'primary_link', 'secondary_link', 'tertiary_link', 'trunk_link', 'motorway', 'motorway_link']
    edges_to_keep = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        hw = data.get('highway')
        if isinstance(hw, list):
            hw = hw[0]
        if hw in valid_highway_types:
            edges_to_keep.append((u, v, k))
            
    filtered_graph = graph.edge_subgraph(edges_to_keep).copy()
    filtered_graph = ox.simplify_graph(filtered_graph)
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
        
    if 'turn:lanes' not in edges.columns:
        edges['turn:lanes'] = "none"
    if 'turn' not in edges.columns:
        edges['turn'] = "none"
        
    def get_first_hw(x):
        return x[0] if isinstance(x, list) else x
    edges['highway'] = edges['highway'].apply(get_first_hw)
    
    # Ensure turn metadata is flattened to strings in case simplification bundled them into lists
    edges['turn:lanes'] = edges['turn:lanes'].apply(get_first_hw).astype(str)
    edges['turn'] = edges['turn'].apply(get_first_hw).astype(str)
    
    if 'oneway' in edges.columns:
        edges['oneway'] = edges['oneway'].apply(get_first_hw).astype(str)
    else:
        edges['oneway'] = "False"
    
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

def get_turn_restrictions(edges_gdf):
    logger.info("Fetching turn restrictions from Overpass API...")
    
    # Ensure the bounding box is in Lat/Lon (EPSG:4326) for the Overpass API
    if edges_gdf.crs != "EPSG:4326":
        minx, miny, maxx, maxy = edges_gdf.to_crs("EPSG:4326").total_bounds
    else:
        minx, miny, maxx, maxy = edges_gdf.total_bounds
        
    overpass_url = "http://overpass-api.de/api/interpreter"
    buffer = 0.001
    overpass_query = f"""
    [out:json][timeout:5];
    relation["type"="restriction"]({miny-buffer},{minx-buffer},{maxy+buffer},{maxx+buffer});
    out;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=5)
        restrictions = []
        if response.status_code == 200:
            data = response.json()
            for element in data.get('elements', []):
                if element['type'] == 'relation':
                    tags = element.get('tags', {})
                    restriction_type = tags.get('restriction', '')
                    from_way = None
                    to_way = None
                    for member in element.get('members', []):
                        if member['role'] == 'from' and member['type'] == 'way':
                            from_way = str(member['ref'])
                        elif member['role'] == 'to' and member['type'] == 'way':
                            to_way = str(member['ref'])
                    if from_way and to_way and restriction_type:
                        restrictions.append({
                            'type': restriction_type,
                            'from': from_way,
                            'to': to_way
                        })
            logger.info(f"Found {len(restrictions)} turn restrictions.")
            return restrictions
        else:
            logger.info(f"Overpass API returned status {response.status_code}.")
    except Exception as e:
        logger.warning(f"Failed to fetch turn restrictions from API: {e}")
        
    logger.info("Falling back to local turn_restrictions.json...")
    if os.path.exists("turn_restrictions.json"):
        try:
            with open("turn_restrictions.json", 'r') as f:
                restrictions = json.load(f)
            logger.info(f"Successfully loaded {len(restrictions)} restrictions from local fallback.")
            return restrictions
        except Exception as e:
            logger.warning(f"Failed to load local fallback: {e}")
    else:
        logger.info("Local fallback turn_restrictions.json not found.")
        
    return []

def parse_osmids(osmid_val):
    if isinstance(osmid_val, list):
        return [str(x) for x in osmid_val]
    elif isinstance(osmid_val, str):
        if osmid_val.startswith('['):
            try:
                return [str(x) for x in ast.literal_eval(osmid_val)]
            except:
                return [str(osmid_val)]
        return [str(osmid_val)]
    return [str(osmid_val)]

def build_topological_adjacency(edges_gdf, target_dist=55.0):
    logger.info("Building topological lookups...")
    # Grouping by node IDs
    edges_by_u = edges_gdf.groupby('u')['segment_id'].apply(list).to_dict()
    edges_by_v = edges_gdf.groupby('v')['segment_id'].apply(list).to_dict()
    
    adjacency = {}
    
    
    restrictions = get_turn_restrictions(edges_gdf)
    
    with open("turn_restrictions.json", "w") as f:
        json.dump(restrictions, f, indent=4)
    logger.info("Saved all extracted turn restrictions to turn_restrictions.json for inspection.")
        
    banned_transitions = []
    only_transitions = []
    
    for r in restrictions:
        rtype = r['type']
        if rtype.startswith('no_'):
            banned_transitions.append((r['from'], r['to'], rtype))
        elif rtype.startswith('only_'):
            only_transitions.append((r['from'], r['to'], rtype))
            
    logger.info("Pre-calculating smoothed headings and lengths...")
    segment_data = {}
    for _, row in edges_gdf.iterrows():
        sid = str(row['segment_id'])
        geom = row['geometry']
        segment_data[sid] = {
            'u': row['u'],
            'v': row['v'],
            'osmid': str(row['osmid']),
            'length': float(row['length']),
            'exit_heading': get_smoothed_heading(geom, reverse=False),
            'entry_heading': get_smoothed_heading(geom, reverse=True)
        }

    logger.info("Analyzing topological connections (choosing best fits)...")
    # First pass: find the single BEST immediate successor and predecessor for every segment
    best_immediate = {}
    for ego_id, data in segment_data.items():
        u_node, v_node = data['u'], data['v']
        ego_osmid = data['osmid']
        ego_osmids = parse_osmids(ego_osmid)
        
        # 1. BEST SUCCESSOR
        best_succ = None
        min_delta_succ = float('inf')
        banned_successors = []
        only_successors = []
        valid_successors = []
        
        only_targets_succ = set()
        for e_id in ego_osmids:
            for (f_id, t_id, rtype) in only_transitions:
                if f_id == e_id:
                    only_targets_succ.add(t_id)
                    
        potential_successors = edges_by_u.get(v_node, [])
        for succ_id in potential_successors:
            sid_str = str(succ_id)
            if sid_str == ego_id: continue
            
            succ_osmids = parse_osmids(segment_data[sid_str]['osmid'])
            succ_heading = segment_data[sid_str]['entry_heading']
            delta_theta = (succ_heading - data['exit_heading'] + 180) % 360 - 180
            
            if -25 <= delta_theta <= 25: direction = 'straight'
            elif 25 < delta_theta < 155: direction = 'left'
            elif -155 < delta_theta < -25: direction = 'right'
            else: direction = 'u_turn'
            
            # Check Turn Restrictions
            is_banned = False
            is_only = False
            
            if only_targets_succ:
                if any(s_id in only_targets_succ for s_id in succ_osmids):
                    is_only = True
                    logger.info(f"Ego [{ego_id}]: Mandatory {direction} transition to {sid_str} confirmed.")
                else:
                    is_banned = True
                    logger.info(f"Ego [{ego_id}]: Connection to {sid_str} filtered out (Only restriction points elsewhere).")
                    
            if not is_banned and not is_only:
                for e_id in ego_osmids:
                    for s_id in succ_osmids:
                        for (f_id, t_id, rtype) in banned_transitions:
                            if f_id == e_id and t_id == s_id:
                                if direction in rtype:
                                    is_banned = True
                                    logger.info(f"Ego [{ego_id}] ({ego_osmids}): Connection to {sid_str} removed by spatial filter ({rtype} explicitly matched {direction}).")
                                    
            if is_banned:
                banned_successors.append(sid_str)
            else:
                valid_successors.append(sid_str)
                
                if is_only:
                    only_successors.append(sid_str)
                    best_succ = sid_str
                    min_delta_succ = -1.0
                else:
                    if min_delta_succ == -1.0: continue
                    
                    is_same_way = any(e_id in succ_osmids for e_id in ego_osmids)
                    threshold = 60.0 if is_same_way else 45.0
                    
                    if abs(delta_theta) <= threshold and abs(delta_theta) < min_delta_succ:
                        min_delta_succ = abs(delta_theta)
                        best_succ = sid_str
        
        # 2. BEST PREDECESSOR
        best_pred = None
        min_delta_pred = float('inf')
        
        only_targets_pred = {}
        for p_id in [str(x) for x in edges_by_v.get(u_node, [])]:
            p_osmids = parse_osmids(segment_data[p_id]['osmid'])
            for po_id in p_osmids:
                for (f_id, t_id, rtype) in only_transitions:
                    if f_id == po_id:
                        only_targets_pred.setdefault(p_id, set()).add(t_id)
                        
        potential_predecessors = edges_by_v.get(u_node, [])
        for pred_id in potential_predecessors:
            sid_str = str(pred_id)
            if sid_str == ego_id: continue
            
            pred_osmids = parse_osmids(segment_data[sid_str]['osmid'])
            pred_heading = segment_data[sid_str]['exit_heading']
            delta_theta = (data['entry_heading'] - pred_heading + 180) % 360 - 180
            
            if -25 <= delta_theta <= 25: direction = 'straight'
            elif 25 < delta_theta < 155: direction = 'left'
            elif -155 < delta_theta < -25: direction = 'right'
            else: direction = 'u_turn'
            
            # Check Turn Restrictions
            is_banned = False
            is_only = False
            
            targets = only_targets_pred.get(sid_str, set())
            if targets:
                if any(e_id in targets for e_id in ego_osmids):
                    is_only = True
                else:
                    is_banned = True
                    
            if not is_banned and not is_only:
                for p_id in pred_osmids:
                    for e_id in ego_osmids:
                        for (f_id, t_id, rtype) in banned_transitions:
                            if f_id == p_id and t_id == e_id:
                                if direction in rtype:
                                    is_banned = True
                                    logger.info(f"Ego [{ego_id}] ({ego_osmids}): Predecessor {sid_str} removed by spatial filter ({rtype} matches {direction}).")
                                    
            if is_banned:
                continue
            
            is_same_way = any(p_id in ego_osmids for p_id in pred_osmids)
            threshold = 60.0 if is_same_way else 45.0
            
            if is_only:
                min_delta_pred = -1.0
                best_pred = sid_str
                break
            
            if abs(delta_theta) <= threshold and abs(delta_theta) < min_delta_pred:
                min_delta_pred = abs(delta_theta)
                best_pred = sid_str
                
        best_immediate[ego_id] = {
            'successor': best_succ, 
            'predecessor': best_pred,
            'banned_successors': banned_successors,
            'only_successors': only_successors,
            'valid_successors': valid_successors
        }

    logger.info(f"Chaining segments to reach {target_dist}m...")
    for ego_id in segment_data.keys():
        # Build Successor Chain
        succ_chain = []
        curr_dist = 0.0
        curr_id = ego_id
        while curr_dist < target_dist:
            next_id = best_immediate[curr_id]['successor']
            if not next_id or next_id in succ_chain or next_id == ego_id:
                break
            succ_chain.append(next_id)
            curr_dist += segment_data[next_id]['length']
            curr_id = next_id
            
        # Build Predecessor Chain
        pred_chain = []
        curr_dist = 0.0
        curr_id = ego_id
        while curr_dist < target_dist:
            prev_id = best_immediate[curr_id]['predecessor']
            if not prev_id or prev_id in pred_chain or prev_id == ego_id:
                break
            pred_chain.append(prev_id)
            curr_dist += segment_data[prev_id]['length']
            curr_id = prev_id
            
        ego_u = segment_data[ego_id]['u']
        ego_v = segment_data[ego_id]['v']
        ego_osmids = parse_osmids(segment_data[ego_id]['osmid'])
        immediate_succ = best_immediate[ego_id]['successor']
        banned_successors = best_immediate[ego_id]['banned_successors']
        only_successors = best_immediate[ego_id]['only_successors']
        valid_successors = best_immediate[ego_id]['valid_successors']
        
        opposite_direction = [str(sid) for sid in edges_by_u.get(ego_v, []) if segment_data[str(sid)]['v'] == ego_u]
        
        succ_opposite = []
        if immediate_succ:
            succ_v = segment_data[immediate_succ]['v']
            succ_opposite = [str(sid) for sid in edges_by_v.get(ego_v, []) if segment_data[str(sid)]['u'] == succ_v]
            
        filtered_intersects_in = [str(sid) for sid in edges_by_v.get(ego_v, []) if str(sid) != ego_id and str(sid) not in opposite_direction and str(sid) not in succ_opposite]
        
        if only_successors:
            filtered_intersects_out = [] # Mandatory transition strips away all other intersection choices.
        else:
            filtered_intersects_out = [str(sid) for sid in valid_successors if str(sid) != immediate_succ and str(sid) not in opposite_direction]

        adjacency[ego_id] = {
            'successors': succ_chain,
            'predecessors': pred_chain,
            'intersects_in': filtered_intersects_in,
            'intersects_out': filtered_intersects_out,
            'banned_successors': banned_successors,
            'only_successors': only_successors,
            'opposite_direction': opposite_direction,
            'successor_lengths': [segment_data[sid]['length'] for sid in succ_chain],
            'predecessor_lengths': [segment_data[sid]['length'] for sid in pred_chain],
            'intersects_in_lengths': [segment_data[sid]['length'] for sid in filtered_intersects_in],
            'intersects_out_lengths': [segment_data[sid]['length'] for sid in filtered_intersects_out],
            'opposite_direction_lengths': [segment_data[sid]['length'] for sid in opposite_direction]
        }
        
    return adjacency

def merge_short_segments(gdf, adj, threshold=15.0):
    logger.info(f"Merging segments shorter than {threshold}m into their neighbors to fill gaps...")
    gdf = gdf.copy()
    gdf.set_index('segment_id', inplace=True, drop=False)
    
    redirect = {}
    def get_target(t):
        while t in redirect:
            t = redirect[t]
        return t

    small_segments = gdf[gdf['length'] < threshold].index.tolist()
    deleted = set()
    
    for s_id in small_segments:
        if s_id in deleted: continue
        if gdf.at[s_id, 'length'] >= threshold: continue
            
        s_adj = adj.get(str(s_id))
        if not s_adj: continue
            
        t_id_original = s_adj['successors'][0] if s_adj.get('successors') else (s_adj['predecessors'][0] if s_adj.get('predecessors') else None)
        is_succ = bool(s_adj.get('successors'))
            
        if not t_id_original: continue
            
        t_id = get_target(t_id_original)
        if t_id == s_id or t_id not in gdf.index: continue
            
        s_row, t_row = gdf.loc[s_id], gdf.loc[t_id]
        
        if is_succ:
            old_node, new_node = s_row['v'], s_row['u']
            s_coords, t_coords = list(s_row['geometry'].coords), list(t_row['geometry'].coords)
            new_coords = s_coords + t_coords[1:] if s_coords[-1] == t_coords[0] else s_coords + t_coords
        else:
            old_node, new_node = s_row['u'], s_row['v']
            s_coords, t_coords = list(s_row['geometry'].coords), list(t_row['geometry'].coords)
            new_coords = t_coords + s_coords[1:] if t_coords[-1] == s_coords[0] else t_coords + s_coords
                
        merged_geom = LineString(new_coords)
        gdf.at[t_id, 'geometry'] = merged_geom
        gdf.at[t_id, 'length'] = merged_geom.length
        
        gdf.loc[gdf['u'] == old_node, 'u'] = new_node
        gdf.loc[gdf['v'] == old_node, 'v'] = new_node
        
        deleted.add(s_id)
        redirect[s_id] = t_id
        
    logger.info(f"Absorbed {len(deleted)} short segments into neighbors.")
    gdf = gdf[~gdf.index.isin(deleted)].copy()
    gdf.reset_index(drop=True, inplace=True)
    gdf['segment_id'] = gdf.index.astype(str)
    return gdf

def main():
    parser = argparse.ArgumentParser(description="Build robust topological adjacency. Always downloads the latest network.")
    parser.add_argument('--output', default='topological_adjacency_test.json', help="Output JSON filename.")
    args = parser.parse_args()
    
    # Always download the network as requested
    edges = get_osm_network()
    
    # 1. Build initial adjacency to know who connects to who
    adjacency = build_topological_adjacency(edges)
    # # 2. Merge short segments to expand neighbors and fill gaps
    # edges = merge_short_segments(edges, adjacency, threshold=15.0)
    # # 3. Rebuild final clean adjacency
    # adjacency = build_topological_adjacency(edges)
    
    network_path = "osm_network_test.gpkg"
    logger.info(f"Saving final merged OSM network to {network_path}...")
    
    # GPKG cannot handle lists well. Convert any list-like osmid or highway tags to simple strings for saving.
    edges_to_save = edges.copy()
    for col in edges_to_save.columns:
        if edges_to_save[col].apply(lambda x: isinstance(x, list)).any():
            edges_to_save[col] = edges_to_save[col].astype(str)
            
    edges_to_save.to_file(network_path, driver="GPKG")

    logger.info(f"Exporting adjacency to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(adjacency, f, indent=4)
    logger.info("Done.")

if __name__ == '__main__':
    main()