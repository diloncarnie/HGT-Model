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
    
    # Propagate 'lanes' values to missing segments along continuous topological paths
    # to allow OSMnx to merge them, while preventing merges of different lane counts.
    changed = True
    iterations = 0
    while changed and iterations < 10:
        changed = False
        iterations += 1
        for node in filtered_graph.nodes():
            in_edges = list(filtered_graph.in_edges(node, keys=True, data=True))
            out_edges = list(filtered_graph.out_edges(node, keys=True, data=True))
            
            # Simple one-way continuation
            if len(in_edges) == 1 and len(out_edges) == 1:
                d1 = in_edges[0][3]
                d2 = out_edges[0][3]
                if 'lanes' in d1 and 'lanes' not in d2:
                    d2['lanes'] = d1['lanes']
                    changed = True
                elif 'lanes' in d2 and 'lanes' not in d1:
                    d1['lanes'] = d2['lanes']
                    changed = True
                    
            # Simple two-way continuation
            elif len(in_edges) == 2 and len(out_edges) == 2:
                in_nodes = {e[0] for e in in_edges}
                out_nodes = {e[1] for e in out_edges}
                
                if in_nodes == out_nodes and len(in_nodes) == 2:
                    n1, n2 = list(in_nodes)
                    
                    for start_n, end_n in [(n1, n2), (n2, n1)]:
                        in_dir = next((e for e in in_edges if e[0] == start_n), None)
                        out_dir = next((e for e in out_edges if e[1] == end_n), None)
                        if in_dir and out_dir:
                            d_in = in_dir[3]
                            d_out = out_dir[3]
                            if 'lanes' in d_in and 'lanes' not in d_out:
                                d_out['lanes'] = d_in['lanes']
                                changed = True
                            elif 'lanes' in d_out and 'lanes' not in d_in:
                                d_in['lanes'] = d_out['lanes']
                                changed = True
                                
            # Cross-intersection propagation for matching roads (based on name or OSM ID)
            if len(in_edges) > 0 and len(out_edges) > 0:
                for _, _, _, d_in in in_edges:
                    for _, _, _, d_out in out_edges:
                        def get_first(val):
                            return val[0] if isinstance(val, list) else val
                            
                        name_in = get_first(d_in.get('name'))
                        name_out = get_first(d_out.get('name'))
                        osmid_in = get_first(d_in.get('osmid'))
                        osmid_out = get_first(d_out.get('osmid'))
                        
                        is_same_name = bool(name_in and name_out and name_in == name_out)
                        is_same_way = bool(osmid_in and osmid_out and str(osmid_in) == str(osmid_out))
                        
                        if is_same_name or is_same_way:
                            if 'lanes' in d_in and 'lanes' not in d_out:
                                d_out['lanes'] = d_in['lanes']
                                changed = True
                            elif 'lanes' in d_out and 'lanes' not in d_in:
                                d_in['lanes'] = d_out['lanes']
                                changed = True
                                
    filtered_graph = ox.simplify_graph(filtered_graph, edge_attrs_differ=['lanes'])
    graph_utm = ox.project_graph(filtered_graph)
    
    nodes, edges = ox.graph_to_gdfs(graph_utm)
    edges = edges.reset_index()
    edges['segment_id'] = edges.index.astype(str)
    
    if 'length' not in edges.columns:
        edges['length'] = edges.geometry.length
    if 'lanes' not in edges.columns:
        edges['lanes'] = np.nan
    else:
        edges['lanes'] = pd.to_numeric(edges['lanes'], errors='coerce')
        
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
            'highway': str(row['highway']),
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
        u_turn_successors = []
        
        only_targets_succ = {}
        for e_id in ego_osmids:
            for (f_id, t_id, rtype) in only_transitions:
                if f_id == e_id:
                    only_targets_succ[t_id] = rtype
                    
        potential_successors = edges_by_u.get(v_node, [])
        
        target_sids = [str(p) for p in potential_successors if any(tid in parse_osmids(segment_data[str(p)]['osmid']) for tid in only_targets_succ.keys())]
        if not target_sids:
            only_targets_succ = {}
            
        for succ_id in potential_successors:
            sid_str = str(succ_id)
            if sid_str == ego_id: continue
            
            succ_osmids = parse_osmids(segment_data[sid_str]['osmid'])
            succ_heading = segment_data[sid_str]['entry_heading']
            delta_theta = (succ_heading - data['exit_heading'] + 180) % 360 - 180
            
            if -25 <= delta_theta <= 25: direction = 'straight'
            elif 25 < delta_theta < 170: direction = 'left'
            elif -170 < delta_theta < -25: direction = 'right'
            else: direction = 'u_turn'
            
            # Apply U-Turn Filters (Undivided Road, Mid-Block, Highway Classification)
            if direction == 'u_turn':
                is_same_way = any(e_id in succ_osmids for e_id in ego_osmids)
                hw_type = segment_data[ego_id]['highway']
                banned_hw = ['primary', 'secondary', 'trunk', 'motorway', 'primary_link', 'secondary_link', 'trunk_link', 'motorway_link']
                is_banned_hw = hw_type in banned_hw
                out_degree = len(potential_successors)
                in_degree = len(edges_by_v.get(v_node, []))
                is_mid_block = (out_degree <= 2 and in_degree <= 2)
                
                if is_same_way or is_banned_hw or is_mid_block:
                    continue  # Invalid structural U-turn, ignore entirely

            # Check Turn Restrictions
            is_banned = False
            is_only = False
            
            if only_targets_succ:
                if any(s_id in only_targets_succ for s_id in succ_osmids):
                    is_only = True
                    matched_rtypes = [only_targets_succ[s_id] for s_id in succ_osmids if s_id in only_targets_succ]
                    logger.info(f"Ego [{ego_id}]: Mandatory {direction} transition to segment {sid_str} confirmed ({matched_rtypes[0]}).")
                else:
                    is_banned = True
                    rtypes = list(set(only_targets_succ.values()))
                    logger.info(f"Ego [{ego_id}] ({ego_osmids}) : Connection to segment {sid_str} filtered out (Only restriction {rtypes} points to segment(s) {target_sids}).")
                    
            if not is_banned and not is_only:
                for e_id in ego_osmids:
                    for s_id in succ_osmids:
                        for (f_id, t_id, rtype) in banned_transitions:
                            if f_id == e_id and t_id == s_id:
                                if direction in rtype:
                                    is_banned = True
                                    logger.info(f"Ego [{ego_id}] ({e_id}): Banned connection to segment {sid_str} removed by spatial filter ({rtype} explicitly matched {direction}).")
                                    
            if is_banned:
                banned_successors.append(sid_str)
            else:
                valid_successors.append(sid_str)
                if direction == 'u_turn':
                    u_turn_successors.append(sid_str)
                    logger.info(f"Ego [{ego_id}]: Valid U-Turn to successor segment {sid_str} detected.")
                
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
                        if p_id not in only_targets_pred:
                            only_targets_pred[p_id] = {}
                        only_targets_pred[p_id][t_id] = rtype
                        
        potential_predecessors = edges_by_v.get(u_node, [])
        for pred_id in potential_predecessors:
            sid_str = str(pred_id)
            if sid_str == ego_id: continue
            
            pred_osmids = parse_osmids(segment_data[sid_str]['osmid'])
            pred_heading = segment_data[sid_str]['exit_heading']
            delta_theta = (data['entry_heading'] - pred_heading + 180) % 360 - 180
            
            if -25 <= delta_theta <= 25: direction = 'straight'
            elif 25 < delta_theta < 170: direction = 'left'
            elif -170 < delta_theta < -25: direction = 'right'
            else: direction = 'u_turn'
            
            # Apply U-Turn Filters for Predecessors
            if direction == 'u_turn':
                is_same_way = any(p_id in ego_osmids for p_id in pred_osmids)
                hw_type = segment_data[ego_id]['highway']
                banned_hw = ['primary', 'secondary', 'trunk', 'motorway', 'primary_link', 'secondary_link', 'trunk_link', 'motorway_link']
                is_banned_hw = hw_type in banned_hw
                out_degree = len(edges_by_u.get(u_node, []))
                in_degree = len(potential_predecessors)
                is_mid_block = (out_degree <= 2 and in_degree <= 2)
                
                if is_same_way or is_banned_hw or is_mid_block:
                    continue  # Invalid structural U-turn, ignore entirely

            # Check Turn Restrictions
            is_banned = False
            is_only = False
            
            targets = only_targets_pred.get(sid_str, {})
            if targets:
                target_sids_pred = [str(p) for p in edges_by_u.get(u_node, []) if any(tid in parse_osmids(segment_data[str(p)]['osmid']) for tid in targets.keys())]
                if not target_sids_pred:
                    targets = {}
                    
            if targets:
                if any(e_id in targets for e_id in ego_osmids):
                    is_only = True
                    matched_rtypes = [targets[e_id] for e_id in ego_osmids if e_id in targets]
                    logger.info(f"Ego [{ego_id}]: Mandatory transition from predecessor segment {sid_str} confirmed ({matched_rtypes[0]}).")
                else:
                    is_banned = True
                    rtypes = list(set(targets.values()))
                    logger.info(f"Ego [{ego_id}] ({ego_osmids}): Predecessor segment {sid_str} filtered out (Only restriction {rtypes} points to segment(s) {target_sids_pred}).")
                    
            if not is_banned and not is_only:
                for p_id in pred_osmids:
                    for e_id in ego_osmids:
                        for (f_id, t_id, rtype) in banned_transitions:
                            if f_id == p_id and t_id == e_id:
                                if direction in rtype:
                                    is_banned = True
                                    logger.info(f"Ego [{ego_id}] ({ego_osmids}): Banned connection from predecessor segment {sid_str} removed by spatial filter ({rtype} explicitly matched {direction}).")
                                    
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
            'valid_successors': valid_successors,
            'u_turn_successors': u_turn_successors
        }

    logger.info(f"Chaining segments to reach {target_dist}m...")
    for ego_id in segment_data.keys():
        # Build Successor Chain
        succ_chain = []
        merging_into = []
        curr_dist = 0.0
        curr_id = ego_id
        while curr_dist < target_dist:
            next_id = best_immediate[curr_id]['successor']
            if not next_id or next_id in succ_chain or next_id == ego_id:
                break
                
            # Merge check: Only consider it a continuation if we are the preferred predecessor
            right_of_way_pred = best_immediate[next_id]['predecessor']
            if right_of_way_pred != curr_id:
                if curr_id == ego_id:
                    merging_into.append(next_id)
                break # We yield right of way, so our topological successor chain ends here
                
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
        immediate_succ = succ_chain[0] if succ_chain else None
        u_turns = best_immediate[ego_id]['u_turn_successors']
        banned_successors = best_immediate[ego_id]['banned_successors']
        only_successors = best_immediate[ego_id]['only_successors']
        valid_successors = best_immediate[ego_id]['valid_successors']
        
        opposite_direction = [str(sid) for sid in edges_by_u.get(ego_v, []) if segment_data[str(sid)]['v'] == ego_u]
        
        succ_opposite = []
        if immediate_succ:
            succ_v = segment_data[immediate_succ]['v']
            succ_opposite = [str(sid) for sid in edges_by_v.get(ego_v, []) if segment_data[str(sid)]['u'] == succ_v]
            
        raw_intersects_in = [str(sid) for sid in edges_by_v.get(ego_v, []) if str(sid) != ego_id and str(sid) not in opposite_direction and str(sid) not in succ_opposite]
        
        filtered_intersects_in = []
        if immediate_succ:
            for sid in raw_intersects_in:
                s_succ = best_immediate[sid]['successor']
                s_only = best_immediate[sid]['only_successors']
                s_valid = best_immediate[sid]['valid_successors']
                
                s_u = segment_data[sid]['u']
                s_v = segment_data[sid]['v']
                s_opposite = [str(x) for x in edges_by_u.get(s_v, []) if segment_data[str(x)]['v'] == s_u]
                
                s_intersects_out = [] if s_only else [str(x) for x in s_valid if str(x) != s_succ and str(x) not in s_opposite]
                
                if immediate_succ == s_succ or immediate_succ in s_intersects_out:
                    filtered_intersects_in.append(sid)
        
        if only_successors:
            filtered_intersects_out = [] # Mandatory transition strips away all other intersection choices.
        else:
            filtered_intersects_out = [str(sid) for sid in valid_successors if str(sid) != immediate_succ and str(sid) not in opposite_direction and str(sid) not in u_turns and str(sid) not in merging_into]

        crosses = []
        targets = set(filtered_intersects_out + merging_into + u_turns)
        for b_id in edges_by_v.get(ego_v, []):
            b_str = str(b_id)
            if b_str != ego_id and best_immediate[b_str]['successor'] in targets:
                if b_str not in crosses:
                    crosses.append(b_str)

        adjacency[ego_id] = {
            'to': succ_chain,
            'from': pred_chain,
            'merges_into': merging_into,
            'crossed_by': filtered_intersects_in,
            'turns_into': filtered_intersects_out,
            'u_turns_into': u_turns,
            'crosses': crosses,
            'banned_successors': banned_successors,
            'only_successors': only_successors,
            'opposite_direction': opposite_direction,
            'to_lengths': [segment_data[sid]['length'] for sid in succ_chain],
            'from_lengths': [segment_data[sid]['length'] for sid in pred_chain],
            'merges_into_lengths': [segment_data[sid]['length'] for sid in merging_into],
            'crossed_by_lengths': [segment_data[sid]['length'] for sid in filtered_intersects_in],
            'turns_into_lengths': [segment_data[sid]['length'] for sid in filtered_intersects_out],
            'u_turns_into_lengths': [segment_data[sid]['length'] for sid in u_turns],
            'crosses_lengths': [segment_data[sid]['length'] for sid in crosses],
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
            
        t_id_original = s_adj['to'][0] if s_adj.get('to') else (s_adj['from'][0] if s_adj.get('from') else None)
        is_succ = bool(s_adj.get('to'))
            
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
    parser.add_argument('--output', default='topological_adjacency.json', help="Output JSON filename.")
    args = parser.parse_args()
    
    # Always download the network as requested
    edges = get_osm_network()
    
    # 1. Build initial adjacency to know who connects to who
    adjacency = build_topological_adjacency(edges)
    # # 2. Merge short segments to expand neighbors and fill gaps
    # edges = merge_short_segments(edges, adjacency, threshold=15.0)
    # # 3. Rebuild final clean adjacency
    # adjacency = build_topological_adjacency(edges)
    
    network_path = "osm_network.gpkg"
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