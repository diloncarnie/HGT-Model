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
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely.strtree import STRtree
import hashlib
import time
import logging

# Set up the Python loggers
def setup_logger(name, log_file):
    l = logging.getLogger(name)
    l.setLevel(logging.INFO)
    if not l.handlers:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        l.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        l.addHandler(ch)
    return l

def cut_line_at_distance(line, distance):
    """
    Cuts a Shapely LineString into two pieces at a specific distance from its start.
    Returns (line_a, line_b).
    """
    if distance <= 0.0:
        return None, line
    if distance >= line.length:
        return line, None
        
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return LineString(coords[:i+1]), LineString(coords[i:])
        if pd > distance:
            cp = line.interpolate(distance)
            return LineString(coords[:i] + [cp]), LineString([cp] + coords[i:])
    return line, None

logger = setup_logger("build_adjacency", "build-adjacency.log")
turn_restrictions_logger = setup_logger("turn_restrictions", "turn_restrictions.log")
junctions_logger = setup_logger("junctions", "junction_detections.log")

def get_osm_network(bbox, config):
    # Instruct OSMnx to retain advanced routing and road rule tags
    ox.settings.useful_tags_way += ['turn:lanes', 'turn', 'junction', 'access']
    ox.settings.useful_tags_node += ['traffic_signals:direction', 'crossing', 'crossing:signals']

    logger.info(f"Downloading OSM network for bounding box: {bbox}...")    # Download using the provided bounding box
    graph = ox.graph_from_bbox(bbox=bbox, network_type='all', simplify=False)

    valid_highway_types = config["valid_highway_types"]
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
    while changed and iterations < config["lane_prop_iterations"]:
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
                                
    # Capture signal nodes with OSM IDs BEFORE simplification. ox.simplify_graph
    # collapses degree-2 nodes, which silently removes signals that lie mid-road on
    # a single way. We keep the OSM node ID for stable controller_id hashing later.
    signal_nodes_raw = []
    for n, d in filtered_graph.nodes(data=True):
        hw = str(d.get('highway', ''))
        crossing = str(d.get('crossing', ''))
        crossing_signals = str(d.get('crossing:signals', '')).lower()
        
        has_hw = 'traffic_signals' in hw
        has_cr = 'traffic_signals' in crossing
        has_cr_sig = crossing_signals in ['yes', 'true', '1']
        
        if has_hw or has_cr or has_cr_sig:
            sig_type = 'highway' if has_hw else 'crossing'
            signal_nodes_raw.append((n, d['x'], d['y'], d.get('traffic_signals:direction'), sig_type))

    filtered_graph = ox.simplify_graph(filtered_graph, edge_attrs_differ=['lanes'])
    graph_utm = ox.project_graph(filtered_graph)

    nodes, edges = ox.graph_to_gdfs(graph_utm)
    edges = edges.reset_index()
    edges['segment_id'] = edges.index.astype(str)

    # Project the pre-simplification signal points into the final edges CRS so the
    # spatial query in tag_signals_at_end compares like-for-like coordinates.
    # Each entry carries its OSM node ID for controller_id hashing downstream.
    if signal_nodes_raw:
        sig_gdf = gpd.GeoDataFrame(
            [{'osm_node_id': int(n), 'ts_direction': str(dir), 'sig_type': str(stype)} for n, _, _, dir, stype in signal_nodes_raw],
            geometry=[Point(lon, lat) for _, lon, lat, _, _ in signal_nodes_raw],
            crs="EPSG:4326"
        ).to_crs(edges.crs)
        
        signal_data = []
        seen_locations = set()
        for idx, row in sig_gdf.iterrows():
            if row.geometry is not None:
                loc_key = (round(row.geometry.x, 5), round(row.geometry.y, 5))
                if loc_key in seen_locations:
                    continue
                seen_locations.add(loc_key)

                signal_data.append({
                    'osm_node_id': int(signal_nodes_raw[idx][0]),
                    'lon': float(signal_nodes_raw[idx][1]),
                    'lat': float(signal_nodes_raw[idx][2]),
                    'ts_direction': signal_nodes_raw[idx][3],
                    'sig_type': signal_nodes_raw[idx][4],
                    'x_utm': float(row.geometry.x),
                    'y_utm': float(row.geometry.y),
                    'geometry': row.geometry
                })
    else:
        signal_data = []
    logger.info(f"Found {len(signal_data)} traffic signal nodes in OSM bbox (pre-simplification).")
    
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
        
    edges['highway'] = edges['highway'].apply(get_first_hw)
    
    if 'junction' not in edges.columns:
        edges['junction'] = "none"
    else:
        edges['junction'] = edges['junction'].fillna("none").apply(get_first_hw).astype(str).str.lower()
        
    logger.info("Initializing internal junction segments...")
    edges['is_internal_junction'] = 'False'

    # Ensure turn metadata is flattened to strings in case simplification bundled them into lists
    edges['turn:lanes'] = edges['turn:lanes'].apply(get_first_hw).astype(str)
    edges['turn'] = edges['turn'].apply(get_first_hw).astype(str)
    
    if 'oneway' in edges.columns:
        edges['oneway'] = edges['oneway'].apply(get_first_hw).astype(str)
    else:
        edges['oneway'] = "False"
    
    # Halve the lanes for two-way segments to accurately represent the individual segment's lane count
    def adjust_lanes(row):
        lanes = row['lanes']
        is_oneway = str(row['oneway']).lower() in ['true', 'yes', '1']
        if pd.notna(lanes) and not is_oneway:
            return max(1, math.ceil(lanes / 2.0))
        return lanes
        
    edges['lanes'] = edges.apply(adjust_lanes, axis=1)

    return edges, signal_data

def tag_signals_at_end(edges, signal_data, adjacency, tolerance=1.0):
    """
    Associates traffic signals to segments in multiple passes to handle overlapping
    or redundant signals effectively:
      1) Assign 'highway' traffic signals first, applying backward filters.
      2) Assign 'crossing' traffic signals, skipping them if they fall within 10m 
         of any existing traffic signal already assigned to the same segment.
      3) Final pass: skip crossing signals that fall within 10m of the segment
         start IF their topological predecessor already has a signal at its end.
         
    Returns (edges, filtered_signal_data) where filtered_signal_data only
    contains the valid signals to be used by the controller builder.
    """
    edges['signal_at_end'] = False
    edges['signal_positions'] = "[]"
    
    if not signal_data:
        return edges, signal_data

    edge_geoms = edges.geometry.tolist()
    edge_sids = edges['segment_id'].astype(str).tolist()
    
    def is_reversed(val):
        if isinstance(val, list):
            val = val[0]
        return str(val).lower() in ['true', 'yes', '1']
        
    edge_revs = [is_reversed(x) for x in edges.get('reversed', pd.Series([False]*len(edges)))]
    edge_oneways = [str(x).lower() in ['true', 'yes', '1'] for x in edges.get('oneway', pd.Series(['False']*len(edges)))]
    edge_lengths = edges['length'].tolist()
    
    edge_tree = STRtree(edge_geoms)
    
    # Store assigned positions for each segment: sid -> [pos, ...]
    edge_signals = {sid: [] for sid in edge_sids}
    valid_signals = []
    
    def get_best_candidate(sig_pt, ts_dir):
        candidate_idxs = edge_tree.query(sig_pt.buffer(tolerance))
        candidates = []
        for edge_idx in candidate_idxs:
            geom = edge_geoms[int(edge_idx)]
            if geom is None or geom.is_empty: continue
            d = geom.distance(sig_pt)
            if d <= tolerance:
                pos = float(geom.project(sig_pt))
                candidates.append({
                    'edge_idx': int(edge_idx),
                    'sid': edge_sids[int(edge_idx)],
                    'dist': d,
                    'pos': pos,
                    'length': edge_lengths[int(edge_idx)],
                    'reversed': edge_revs[int(edge_idx)],
                    'is_oneway': edge_oneways[int(edge_idx)]
                })
                
        best_candidate = None
        best_score = -float('inf')
        for c in candidates:
            # Apply backward filter for oneway segments
            if 'backward' in ts_dir and c['is_oneway']:
                continue
            # Apply directional filters for two-way segments
            if not c['is_oneway'] and ts_dir and ts_dir not in ['none', 'nan', 'null', '']:
                if 'forward' in ts_dir and c['reversed']:
                    continue
                if 'backward' in ts_dir and not c['reversed']:
                    continue
                    
            score = -c['dist'] * 10.0
            pos_ratio = c['pos'] / c['length'] if c['length'] > 0 else 0.5
            score += pos_ratio * 50.0 
            
            if score > best_score:
                best_score = score
                best_candidate = c
                
        return best_candidate

    # Pass 1: Assign highway=traffic_signal nodes
    highway_signals = [(idx, s) for idx, s in enumerate(signal_data) if str(s.get('sig_type')) == 'highway']
    crossing_signals = [(idx, s) for idx, s in enumerate(signal_data) if str(s.get('sig_type')) == 'crossing']
    
    for idx, s in highway_signals:
        sig_pt = s['geometry']
        ts_dir = str(s.get('ts_direction', '')).lower()
        
        best_candidate = get_best_candidate(sig_pt, ts_dir)
        if best_candidate is None:
            if 'backward' in ts_dir:
                logger.info(
                    f"Skipped backward highway signal (OSM node {s['osm_node_id']}) "
                    f"as no valid segment was found"
                )
            continue
            
        s['hosted_on_segment'] = best_candidate['sid']
        s['position_along_host_m'] = best_candidate['pos']
        valid_signals.append(s)
        edge_signals[best_candidate['sid']].append(best_candidate['pos'])

    # Lookup to track segments with signals from Pass 1
    signal_at_end_lookup = {sid: bool(pos_list) for sid, pos_list in edge_signals.items()}

    # Pass 2: Assign crossing=traffic_signal nodes
    for idx, s in crossing_signals:
        sig_pt = s['geometry']
        ts_dir = str(s.get('ts_direction', '')).lower()
        
        best_candidate = get_best_candidate(sig_pt, ts_dir)
        if best_candidate is None:
            if 'backward' in ts_dir:
                logger.info(
                    f"Skipped backward crossing signal (OSM node {s['osm_node_id']}) "
                    f"as no valid segment was found"
                )
            continue
            
        sid = best_candidate['sid']
        pos = best_candidate['pos']
        length = best_candidate['length']
        
        too_close = False
        for p in edge_signals.get(sid, []):
            if abs(pos - p) <= 15.0:
                too_close = True
                break
        
        # Skip if this crossing signal is within 10m of an already assigned signal
        if too_close:
            logger.info(
                f"Skipped crossing signal (OSM node {s['osm_node_id']}) "
                f"as it is within 10m of an existing signal on segment {sid}."
            )
            continue
            
        s['hosted_on_segment'] = sid
        s['position_along_host_m'] = pos
        valid_signals.append(s)
        edge_signals[sid].append(pos)
        signal_at_end_lookup[sid] = True

    # Pass 3: Check for Invalid Crossing traffic signals as the final pass.
    final_valid_signals = []
    for s in valid_signals:
        if str(s.get('sig_type')) == 'crossing':
            sid = s.get('hosted_on_segment')
            pos = s.get('position_along_host_m')
            
            if sid and pos is not None and pos <= 15.0:
                is_invalid = False
                if sid in adjacency:
                    for pred_sid in adjacency[sid].get('from', []):
                        if signal_at_end_lookup.get(str(pred_sid), False):
                            is_invalid = True
                            break
                            
                if is_invalid:
                    logger.info(
                        f"Skipped invalid crossing signal (OSM node {s['osm_node_id']}) "
                        f"as it is close to start of {sid} which already has a signal at the end of its predecessor."
                    )
                    # Remove it from edge_signals
                    if pos in edge_signals[sid]:
                        edge_signals[sid].remove(pos)
                        if not edge_signals[sid]:
                            signal_at_end_lookup[sid] = False
                    continue
                    
        final_valid_signals.append(s)

    # Write back to edges dataframe
    sig_at_end_flags = []
    sig_positions_list = []
    n_end = 0
    n_mid = 0
    
    for idx, row in edges.iterrows():
        sid = str(row['segment_id'])
        positions = edge_signals.get(sid, [])
        positions.sort()
        
        sig_positions_list.append(json.dumps(positions))
        
        has_any = bool(positions)
        sig_at_end_flags.append(has_any)
        
        if has_any:
            # Check if at terminal vertex
            geom = row.geometry
            if geom and not geom.is_empty:
                end_pt = Point(list(geom.coords)[-1])
                is_terminal = False
                for p in positions:
                    if geom.length - p <= tolerance:
                        is_terminal = True
                        break
                if is_terminal:
                    n_end += 1
                else:
                    n_mid += 1

    edges['signal_at_end'] = sig_at_end_flags
    edges['signal_positions'] = sig_positions_list
    
    logger.info(
        f"Tagged {n_end} edges with signal_at_end=True and {n_mid} additional "
        f"edges with mid-segment signals (tolerance {tolerance} m)."
    )

    return edges, final_valid_signals


def build_controllers(edges, signal_data, adjacency, config):
    """Build one logical signal controller per signalized junction.

    Produces a ``controllers`` dict keyed by a stable ``controller_id``
    (sha1-12 of sorted OSM signal node IDs for junction controllers, or
    ``s_<osm_node_id>`` for standalone mid-block signals) and writes
    ``controller_id`` / ``controller_role`` columns back onto *edges*.

    Returns ``(controllers_dict, edges)``.
    """
    import networkx as nx
    from shapely.ops import polygonize
    from shapely.geometry import MultiPoint

    tolerance = config.get("signal_endpoint_tolerance", 1.0)
    junction_tolerance = config.get("controller_junction_tolerance", 5.0)

    logger.info("Building signal controllers...")

    # Initialize output columns
    edges = edges.copy()
    edges['controller_id'] = ""
    edges['controller_role'] = ""

    if not signal_data:
        logger.info("No signal data available; skipping controller construction.")
        return {}, edges

    # ------------------------------------------------------------------
    # Step 2  Attach each signal to its hosted segment
    # ------------------------------------------------------------------
    signal_points = [s['geometry'] for s in signal_data]
    edge_geoms = edges.geometry.tolist()
    edge_sids = edges['segment_id'].astype(str).tolist()
    
    signal_at_end_lookup = {
        str(sid): str(val).lower() == 'true' 
        for sid, val in zip(edges['segment_id'], edges.get('signal_at_end', False))
    }
    
    def is_reversed(val):
        if isinstance(val, list):
            val = val[0]
        return str(val).lower() in ['true', 'yes', '1']
        
    edge_revs = [is_reversed(x) for x in edges['reversed'].tolist()] if 'reversed' in edges.columns else [False] * len(edges)
    edge_oneways = [str(x).lower() in ['true', 'yes', '1'] for x in edges['oneway'].tolist()]
    edge_lengths = edges['length'].tolist()
    
    edge_tree = STRtree(edge_geoms)

    for sig_idx, sig_pt in enumerate(signal_points):
        candidate_idxs = edge_tree.query(sig_pt.buffer(tolerance))
        
        candidates = []
        for edge_idx in candidate_idxs:
            geom = edge_geoms[int(edge_idx)]
            if geom is None or geom.is_empty:
                continue
            d = geom.distance(sig_pt)
            if d <= tolerance:
                pos = float(geom.project(sig_pt))
                candidates.append({
                    'edge_idx': int(edge_idx),
                    'sid': edge_sids[int(edge_idx)],
                    'dist': d,
                    'pos': pos,
                    'length': edge_lengths[int(edge_idx)],
                    'reversed': edge_revs[int(edge_idx)],
                    'is_oneway': edge_oneways[int(edge_idx)]
                })
                
        if not candidates:
            signal_data[sig_idx]['hosted_on_segment'] = None
            signal_data[sig_idx]['position_along_host_m'] = None
            continue
            
        ts_dir = str(signal_data[sig_idx].get('ts_direction', '')).lower()
        
        best_candidate = None
        best_score = -float('inf')
        
        for c in candidates:
            # First apply the backward filter on oneway=true segments
            if 'backward' in ts_dir and c['is_oneway']:
                continue
                
            # Then assign traffic lights on oneway=false segments based on their direction
            if not c['is_oneway'] and ts_dir and ts_dir not in ['none', 'nan', 'null', '']:
                if 'forward' in ts_dir and c['reversed']:
                    continue
                if 'backward' in ts_dir and not c['reversed']:
                    continue

            score = -c['dist'] * 10.0
            
            # Heuristic: traffic lights are near the END of the segment (entering intersection)
            if c['length'] > 0:
                pos_ratio = c['pos'] / c['length']
            else:
                pos_ratio = 0.5
                
            score += pos_ratio * 50.0 
            
            if score > best_score:
                best_score = score
                best_candidate = c
                
        if best_candidate is None:
            signal_data[sig_idx]['hosted_on_segment'] = None
            signal_data[sig_idx]['position_along_host_m'] = None
            if 'backward' in ts_dir:
                logger.info(
                    f"Skipped backward signal {sig_idx} (OSM node "
                    f"{signal_data[sig_idx]['osm_node_id']}) as no valid segment was found"
                )
            continue

        signal_data[sig_idx]['hosted_on_segment'] = best_candidate['sid']
        signal_data[sig_idx]['position_along_host_m'] = best_candidate['pos']

    # segment -> [signal indices]
    seg_to_signals = {}
    for sig_idx, sd in enumerate(signal_data):
        host = sd.get('hosted_on_segment')
        if host:
            seg_to_signals.setdefault(host, []).append(sig_idx)

    # ------------------------------------------------------------------
    # Step 3  Identify signalized junctions via junction groupings
    # ------------------------------------------------------------------
    junction_mask = edges['is_internal_junction'].astype(str).str.lower() == 'true'
    junction_edges_gdf = edges[junction_mask]

    # Quick segment lookup
    seg_lookup = {}
    for _, row in edges.iterrows():
        seg_lookup[str(row['segment_id'])] = row

    # Connected components of internal-junction edges
    G_junc = nx.MultiGraph()
    for _, row in junction_edges_gdf.iterrows():
        sid = str(row['segment_id'])
        G_junc.add_edge(row['u'], row['v'], key=sid, segment_id=sid)

    junction_groups = {}
    seg_to_group = {}
    for group_id, component in enumerate(nx.connected_components(G_junc)):
        subgraph = G_junc.subgraph(component)
        segment_ids = list({data['segment_id']
                           for _, _, _, data in subgraph.edges(keys=True, data=True)})
        junction_groups[group_id] = {
            'junction_segments': segment_ids,
            'nodes': set(component),
        }
        for sid in segment_ids:
            seg_to_group[sid] = group_id

    # Build junction polygons for spatial approach-segment detection
    seen_pairs = set()
    unique_junction_geoms = []
    for _, row in junction_edges_gdf.iterrows():
        key = frozenset((row['u'], row['v']))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_junction_geoms.append(row.geometry)

    junction_polys = list(polygonize(unique_junction_geoms))

    # Assign each polygon to its junction group
    group_polys = {}
    for poly in junction_polys:
        for group_id, group in junction_groups.items():
            matched = False
            for sid in group['junction_segments']:
                row = seg_lookup.get(sid)
                if row is not None and row.geometry is not None:
                    if poly.boundary.intersection(row.geometry).length > 1e-5:
                        if group_id not in group_polys:
                            group_polys[group_id] = poly
                        else:
                            group_polys[group_id] = group_polys[group_id].union(poly)
                        matched = True
                        break
            if matched:
                break

    # Fallback geometry for groups without polygons
    for group_id, group in junction_groups.items():
        if group_id not in group_polys:
            node_coords = []
            for sid in group['junction_segments']:
                row = seg_lookup.get(sid)
                if row is not None and row.geometry is not None:
                    node_coords.extend(list(row.geometry.coords))
            if len(node_coords) >= 3:
                hull = MultiPoint(node_coords).convex_hull
                group_polys[group_id] = hull if hull.area > 0 else hull.buffer(junction_tolerance)
            elif node_coords:
                group_polys[group_id] = MultiPoint(node_coords).convex_hull.buffer(junction_tolerance)

    # ------------------------------------------------------------------
    # Step 4  Find approach segments for each junction group
    # ------------------------------------------------------------------
    non_junction_edges = edges[~junction_mask]

    # Pre-index: v-node -> non-junction segment IDs
    v_to_non_junc_segs = {}
    v_node_points = {}
    for _, row in non_junction_edges.iterrows():
        sid = str(row['segment_id'])
        v_node = row['v']
        v_to_non_junc_segs.setdefault(v_node, []).append(sid)
        if v_node not in v_node_points and row.geometry is not None and not row.geometry.is_empty:
            v_node_points[v_node] = Point(list(row.geometry.coords)[-1])

    for group_id, group in junction_groups.items():
        approach = set()
        # Topological: v node is a junction-group node
        for v_node in group['nodes']:
            for sid in v_to_non_junc_segs.get(v_node, []):
                approach.add(sid)
        # Spatial: v point is within tolerance of junction polygon
        group_poly = group_polys.get(group_id)
        if group_poly is not None:
            for v_node, v_pt in v_node_points.items():
                if v_node not in group['nodes'] and group_poly.distance(v_pt) <= junction_tolerance:
                    for sid in v_to_non_junc_segs.get(v_node, []):
                        approach.add(sid)
        group['approach_segments'] = list(approach)

    # ------------------------------------------------------------------
    # Step 4b  Reassign long boundary edges back to junction segments
    # ------------------------------------------------------------------
    # classify_junctions marks edges that lie on a junction polygon boundary
    # but were excluded because their length exceeded classify_shape_max_length_m
    # (is_long_shape_boundary=True). For shape-junction groups, any approach
    # segment carrying that flag is actually part of the junction polygon and
    # should be a junction link. We reassign it unless was_merged=True, which
    # indicates the segment absorbed a neighbour in merge_short_segments or
    # simplify_network_topology and should not be treated as a pure junction link.
    newly_assigned_junction_sids = set()

    has_long_boundary_col = 'is_long_shape_boundary' in edges.columns
    has_merged_col = 'was_merged' in edges.columns

    for group_id, group in junction_groups.items():
        # Only consider groups built from shape-junction segments
        has_shape = any(
            str(seg_lookup[sid].get('is_shape_junction', 'False')).lower() == 'true'
            for sid in group['junction_segments']
            if sid in seg_lookup
        )
        if not has_shape:
            continue

        for sid in list(group['approach_segments']):
            row = seg_lookup.get(sid)
            if row is None:
                continue
            if not has_long_boundary_col or str(row.get('is_long_shape_boundary', 'False')).lower() != 'true':
                continue
            if str(row.get('is_internal_junction', 'False')).lower() == 'true':
                continue
            if has_merged_col and str(row.get('was_merged', 'False')).lower() == 'true':
                continue

            # Only reassign if the candidate actually closes the junction shape.
            # Combine the group's existing junction segment geometries with the
            # candidate and check whether polygonize produces a closed polygon.
            seen_pairs = set()
            geoms_without_candidate = []
            for jsid in group['junction_segments']:
                jrow = seg_lookup.get(jsid)
                if jrow is None:
                    continue
                key = frozenset((jrow['u'], jrow['v']))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if jrow.geometry is not None and not jrow.geometry.is_empty:
                    geoms_without_candidate.append(jrow.geometry)
            
            polys_without = list(polygonize(geoms_without_candidate))

            cand_key = frozenset((row['u'], row['v']))
            geoms_to_test = list(geoms_without_candidate)
            if cand_key not in seen_pairs and row.geometry is not None and not row.geometry.is_empty:
                geoms_to_test.append(row.geometry)
            
            polys_with = list(polygonize(geoms_to_test))

            if len(polys_with) <= len(polys_without):
                continue

            newly_assigned_junction_sids.add(sid)
            group['junction_segments'].append(sid)
            group['approach_segments'].remove(sid)
            seg_to_group[sid] = group_id
            junctions_logger.info(
                f"Reassigned long boundary segment {sid} "
                f"(length {float(row.get('length', 0)):.1f}m) as junction link "
                f"for group {group_id} (closes shape)."
            )

    if newly_assigned_junction_sids:
        mask_new = edges['segment_id'].astype(str).isin(newly_assigned_junction_sids)
        edges.loc[mask_new, 'is_internal_junction'] = 'True'
        logger.info(
            f"Reassigned {len(newly_assigned_junction_sids)} long boundary edge(s) "
            f"as junction segments for shape-junction groups."
        )

    # Determine which groups are signalized
    signalized_groups = {}
    for group_id, group in junction_groups.items():
        all_segs = set(group['junction_segments']) | set(group.get('approach_segments', []))
        sig_indices = []
        for sid in all_segs:
            sigs = seg_to_signals.get(sid, [])
            if not sigs:
                continue
            
            # Skip _link roads with exactly one signal so they become standalone controllers
            row = seg_lookup.get(sid)
            if row is not None and str(row.get('highway', '')).endswith('_link') and len(sigs) == 1:
                continue
                
            sig_indices.extend(sigs)
        if sig_indices:
            signalized_groups[group_id] = sig_indices

    # ------------------------------------------------------------------
    # Step 4b  Form controllers from signalized groups
    # ------------------------------------------------------------------
    controllers = {}
    used_signals = set()

    for group_id, sig_indices in signalized_groups.items():
        group = junction_groups[group_id]

        if len(sig_indices) == 1:
            junctions_logger.info(f"Disqualified junction group {group_id} as a controller because it only has 1 signal.")
            continue

        raw_osm_signals = []
        osm_node_ids = []
        for si in sig_indices:
            sd = signal_data[si]
            raw_osm_signals.append({
                'osm_node_id': sd['osm_node_id'],
                'lon': sd['lon'], 'lat': sd['lat'],
                'x_utm': sd['x_utm'], 'y_utm': sd['y_utm'],
                'hosted_on_segment': sd.get('hosted_on_segment'),
                'position_along_host_m': sd.get('position_along_host_m'),
            })
            osm_node_ids.append(sd['osm_node_id'])
            used_signals.add(si)

        # Stable controller_id: sha1-12 of sorted OSM signal node IDs
        sorted_ids = sorted(set(osm_node_ids))
        id_str = ','.join(str(x) for x in sorted_ids)
        controller_id = hashlib.sha1(id_str.encode()).hexdigest()[:12]

        # Junction centroid
        group_poly = group_polys.get(group_id)
        if group_poly is not None:
            centroid_utm = group_poly.centroid
        else:
            xs = [signal_data[i]['x_utm'] for i in sig_indices]
            ys = [signal_data[i]['y_utm'] for i in sig_indices]
            centroid_utm = Point(np.mean(xs), np.mean(ys))

        centroid_ll = gpd.GeoDataFrame(
            geometry=[centroid_utm], crs=edges.crs
        ).to_crs("EPSG:4326").geometry.iloc[0]

        junction_centroid = {
            'lon': centroid_ll.x, 'lat': centroid_ll.y,
            'x_utm': centroid_utm.x, 'y_utm': centroid_utm.y,
        }

        # junction_id: pick a shape-junction segment or first junction segment
        junction_id = None
        for sid in group['junction_segments']:
            row = seg_lookup.get(sid)
            if row is not None and str(row.get('is_shape_junction', 'False')).lower() == 'true':
                junction_id = sid
                break
        if junction_id is None and group['junction_segments']:
            junction_id = group['junction_segments'][0]

        controllers[controller_id] = {
            'controller_id': controller_id,
            'junction_id': junction_id,
            'approach_segments': group.get('approach_segments', []),
            'junction_segments': group['junction_segments'],
            'raw_osm_signals': raw_osm_signals,
            'signal_count': len(raw_osm_signals),
            'junction_centroid': junction_centroid,
            'source': 'junction_polygon',
        }

    # ------------------------------------------------------------------
    # Step 5  Standalone signals (not absorbed by any junction)
    # ------------------------------------------------------------------
    standalone_groups = {}
    ungrouped_signals = []

    for sig_idx, sd in enumerate(signal_data):
        if sig_idx in used_signals:
            continue
        
        host_seg = sd.get('hosted_on_segment')
        if host_seg and host_seg in seg_lookup:
            row = seg_lookup[host_seg]
            sigs = seg_to_signals.get(host_seg, [])
            if str(row.get('highway', '')).endswith('_link') and len(sigs) == 1:
                ungrouped_signals.append((sig_idx, sd))
            else:
                v_node = row['v']
                standalone_groups.setdefault(v_node, []).append((sig_idx, sd))
        else:
            ungrouped_signals.append((sig_idx, sd))

    for v_node, sig_list in standalone_groups.items():
        if len(sig_list) == 1:
            sig_idx, sd = sig_list[0]
            osm_node_id = sd['osm_node_id']
            controller_id = f"s_{osm_node_id}"
            host_seg = sd.get('hosted_on_segment')
            
            # For standalone signal groups containing only 1 signal, restrict approach_segments exclusively to the host_seg.
            # This prevents unrelated segments (e.g. crossing roads sharing the node) from being added.
            approach_set = {host_seg} if host_seg else set()
            approach_segments = list(approach_set)

            raw_osm_signals = [{
                'osm_node_id': sd['osm_node_id'],
                'lon': sd['lon'], 'lat': sd['lat'],
                'x_utm': sd['x_utm'], 'y_utm': sd['y_utm'],
                'hosted_on_segment': host_seg,
                'position_along_host_m': sd.get('position_along_host_m'),
            }]

            sig_pt_ll = gpd.GeoDataFrame(
                geometry=[sd['geometry']], crs=edges.crs
            ).to_crs("EPSG:4326").geometry.iloc[0]

            junction_centroid = {
                'lon': sig_pt_ll.x, 'lat': sig_pt_ll.y,
                'x_utm': sd['x_utm'], 'y_utm': sd['y_utm'],
            }

            controllers[controller_id] = {
                'controller_id': controller_id,
                'junction_id': None,
                'approach_segments': approach_segments,
                'junction_segments': [],
                'raw_osm_signals': raw_osm_signals,
                'signal_count': 1,
                'junction_centroid': junction_centroid,
                'source': 'standalone_signal',
            }
        else:
            osm_node_ids = [sd['osm_node_id'] for _, sd in sig_list]
            sorted_ids = sorted(set(osm_node_ids))
            id_str = ','.join(str(x) for x in sorted_ids)
            controller_id = hashlib.sha1(id_str.encode()).hexdigest()[:12]
            
            approach_set = set()
            raw_osm_signals = []
            xs, ys = [], []
            for _, sd in sig_list:
                host_seg = sd.get('hosted_on_segment')
                if host_seg:
                    approach_set.add(host_seg)
                    if host_seg in adjacency:
                        adj_ego = adjacency[host_seg]
                        # Discard merge segments
                        for s in adj_ego.get('merges_with', []):
                            approach_set.discard(s)
                        for s in adj_ego.get('merged_by', []):
                            approach_set.discard(s)

                raw_osm_signals.append({
                    'osm_node_id': sd['osm_node_id'],
                    'lon': sd['lon'], 'lat': sd['lat'],
                    'x_utm': sd['x_utm'], 'y_utm': sd['y_utm'],
                    'hosted_on_segment': host_seg,
                    'position_along_host_m': sd.get('position_along_host_m'),
                })
                xs.append(sd['x_utm'])
                ys.append(sd['y_utm'])
            approach_segments = list(approach_set)
            
            centroid_utm = Point(np.mean(xs), np.mean(ys))
            centroid_ll = gpd.GeoDataFrame(
                geometry=[centroid_utm], crs=edges.crs
            ).to_crs("EPSG:4326").geometry.iloc[0]

            junction_centroid = {
                'lon': centroid_ll.x, 'lat': centroid_ll.y,
                'x_utm': centroid_utm.x, 'y_utm': centroid_utm.y,
            }

            controllers[controller_id] = {
                'controller_id': controller_id,
                'junction_id': None,
                'approach_segments': approach_segments,
                'junction_segments': [],
                'raw_osm_signals': raw_osm_signals,
                'signal_count': len(raw_osm_signals),
                'junction_centroid': junction_centroid,
                'source': 'standalone_signal',
            }

    for sig_idx, sd in ungrouped_signals:
        osm_node_id = sd['osm_node_id']
        controller_id = f"s_{osm_node_id}"

        host_seg = sd.get('hosted_on_segment')
        approach_set = {host_seg} if host_seg else set()
        if host_seg and host_seg in adjacency:
            adj_ego = adjacency[host_seg]
            # Discard merge segments
            for s in adj_ego.get('merges_with', []):
                approach_set.discard(s)
            for s in adj_ego.get('merged_by', []):
                approach_set.discard(s)
            approach_segments = list(approach_set)
        raw_osm_signals = [{
            'osm_node_id': sd['osm_node_id'],
            'lon': sd['lon'], 'lat': sd['lat'],
            'x_utm': sd['x_utm'], 'y_utm': sd['y_utm'],
            'hosted_on_segment': host_seg,
            'position_along_host_m': sd.get('position_along_host_m'),
        }]

        sig_pt_ll = gpd.GeoDataFrame(
            geometry=[sd['geometry']], crs=edges.crs
        ).to_crs("EPSG:4326").geometry.iloc[0]

        junction_centroid = {
            'lon': sig_pt_ll.x, 'lat': sig_pt_ll.y,
            'x_utm': sd['x_utm'], 'y_utm': sd['y_utm'],
        }

        controllers[controller_id] = {
            'controller_id': controller_id,
            'junction_id': None,
            'approach_segments': approach_segments,
            'junction_segments': [],
            'raw_osm_signals': raw_osm_signals,
            'signal_count': 1,
            'junction_centroid': junction_centroid,
            'source': 'standalone_signal',
        }

    # ------------------------------------------------------------------
    # Step 5b Filter link roads from approach segments
    # ------------------------------------------------------------------
    for ctrl_id, ctrl in controllers.items():
        occupied_segments = {s.get('hosted_on_segment') for s in ctrl['raw_osm_signals']}
        valid_approaches = []
        for sid in ctrl['approach_segments']:
            row = seg_lookup.get(sid)
            if row is not None and str(row.get('highway', '')).endswith('_link'):
                if sid in occupied_segments:
                    valid_approaches.append(sid)
            else:
                valid_approaches.append(sid)
        
        ctrl['approach_segments'] = valid_approaches

    # ------------------------------------------------------------------
    # Step 5c Convert short approach segments to junction links
    # ------------------------------------------------------------------
    logger.info("Checking for short approach segments (< 25m) to convert to junction links...")
    short_approach_threshold = 25.0
    converted_count = 0

    for ctrl_id, ctrl in controllers.items():
        changed = True
        while changed:
            changed = False
            new_approaches = set(ctrl.get('approach_segments', []))
            for sid in list(new_approaches):
                row = seg_lookup.get(sid)
                if row is not None and 'length' in row and float(row['length']) < short_approach_threshold:
                    new_approaches.remove(sid)
                    if sid not in ctrl['junction_segments']:
                        ctrl['junction_segments'].append(sid)
                        
                    # Mark as junction link
                    mask = edges['segment_id'].astype(str) == sid
                    edges.loc[mask, 'is_internal_junction'] = 'True'
                    
                    # Add predecessors to approaches
                    if sid in adjacency:
                        for pred in adjacency[sid].get('from', []):
                            if pred not in ctrl['junction_segments']:
                                new_approaches.add(pred)
                    
                    changed = True
                    converted_count += 1
                    junctions_logger.info(f"Converted short approach segment {sid} (< 25m) to junction link for controller {ctrl_id}.")
                    
            ctrl['approach_segments'] = list(new_approaches)
            
    if converted_count > 0:
        logger.info(f"Converted {converted_count} short approach segments into junction links.")

    # ------------------------------------------------------------------
    # Step 5d Recover approach segments for isolated standalone controllers
    # ------------------------------------------------------------------
    for ctrl_id, ctrl in controllers.items():
        if ctrl.get('source') == 'standalone_signal' and not ctrl.get('approach_segments') and ctrl.get('junction_segments'):
            recovered_approaches = set()
            for sid in ctrl['junction_segments']:
                if sid in adjacency:
                    for succ in adjacency[sid].get('to', []):
                        turns_cands = []
                        merges_cands = []
                        for cand_id, cand_adj in adjacency.items():
                            if cand_id == sid:
                                continue
                            if succ in cand_adj.get('turns_into', []):
                                turns_cands.append(cand_id)
                            if succ in cand_adj.get('merges_into', []):
                                merges_cands.append(cand_id)
                        
                        if turns_cands:
                            recovered_approaches.update(turns_cands)
                        elif merges_cands:
                            recovered_approaches.update(merges_cands)
            
            for cand in recovered_approaches:
                if cand not in ctrl['junction_segments']:
                    ctrl['approach_segments'].append(cand)
                    junctions_logger.info(f"Recovered approach segment {cand} for isolated standalone controller {ctrl_id}.")

    # ------------------------------------------------------------------
    # Step 5e Merge standalone controllers sharing the exact same single approach segment
    # ------------------------------------------------------------------
    approach_to_ctrls = {}
    for ctrl_id, ctrl in controllers.items():
        if ctrl.get('source') == 'standalone_signal' and len(ctrl.get('approach_segments', [])) == 1:
            app_sid = ctrl['approach_segments'][0]
            approach_to_ctrls.setdefault(app_sid, []).append(ctrl_id)
            
    for app_sid, ctrl_ids in approach_to_ctrls.items():
        if len(ctrl_ids) > 1:
            raw_osm_signals = []
            junction_segs = set()
            osm_node_ids = []
            xs = []
            ys = []
            
            for cid in ctrl_ids:
                ctrl = controllers[cid]
                raw_osm_signals.extend(ctrl['raw_osm_signals'])
                junction_segs.update(ctrl.get('junction_segments', []))
                for sig in ctrl['raw_osm_signals']:
                    osm_node_ids.append(sig['osm_node_id'])
                    xs.append(sig['x_utm'])
                    ys.append(sig['y_utm'])
                    
            sorted_ids = sorted(set(osm_node_ids))
            id_str = ','.join(str(x) for x in sorted_ids)
            new_controller_id = hashlib.sha1(id_str.encode()).hexdigest()[:12]
            
            centroid_utm = Point(np.mean(xs), np.mean(ys))
            centroid_ll = gpd.GeoDataFrame(
                geometry=[centroid_utm], crs=edges.crs
            ).to_crs("EPSG:4326").geometry.iloc[0]

            junction_centroid = {
                'lon': centroid_ll.x, 'lat': centroid_ll.y,
                'x_utm': centroid_utm.x, 'y_utm': centroid_utm.y,
            }
            
            new_ctrl = {
                'controller_id': new_controller_id,
                'junction_id': None,
                'approach_segments': [app_sid],
                'junction_segments': list(junction_segs),
                'raw_osm_signals': raw_osm_signals,
                'signal_count': len(raw_osm_signals),
                'junction_centroid': junction_centroid,
                'source': 'standalone_signal',
            }
            
            for cid in ctrl_ids:
                del controllers[cid]
                
            controllers[new_controller_id] = new_ctrl
            junctions_logger.info(f"Merged {len(ctrl_ids)} standalone controllers sharing single approach {app_sid} into new controller {new_controller_id}.")

    # ------------------------------------------------------------------
    # Step 6  Write back to edges
    # ------------------------------------------------------------------
    seg_to_controller = {}
    seg_to_role = {}
    for ctrl_id, ctrl in controllers.items():
        for sid in ctrl['approach_segments']:
            seg_to_controller[sid] = ctrl_id
            seg_to_role[sid] = 'approach'
        for sid in ctrl['junction_segments']:
            # Approach role takes precedence if a segment appears in both
            if sid not in seg_to_controller:
                seg_to_controller[sid] = ctrl_id
                seg_to_role[sid] = 'junction_internal'

    ctrl_ids = []
    ctrl_roles = []
    sig_at_end = []
    for _, row in edges.iterrows():
        sid = str(row['segment_id'])
        cid = seg_to_controller.get(sid, "")
        role = seg_to_role.get(sid, "")
        ctrl_ids.append(cid)
        ctrl_roles.append(role)
        # Redefined: True iff the segment has a controller and is an approach
        sig_at_end.append(bool(cid and role == 'approach'))

    edges['controller_id'] = ctrl_ids
    edges['controller_role'] = ctrl_roles
    edges['signal_at_end'] = sig_at_end

    n_junc = sum(1 for c in controllers.values() if c['source'] == 'junction_polygon')
    n_solo = sum(1 for c in controllers.values() if c['source'] == 'standalone_signal')
    n_approach = sum(len(c['approach_segments']) for c in controllers.values())
    n_internal = sum(len(c['junction_segments']) for c in controllers.values())
    logger.info(
        f"Built {len(controllers)} signal controllers "
        f"({n_junc} junction-based, {n_solo} standalone)."
    )
    logger.info(f"  Approach segments: {n_approach}, Junction-internal segments: {n_internal}")

    return controllers, edges


def split_multi_signal_segments(edges, adjacency, config):
    """
    Splits one-way segments at the first traffic signal when:
      - the segment has multiple signals (split 3.5m past the first), or
      - the segment has a single signal within 10m of its start (split 3.5m
        past that signal so the short piece becomes a junction link via
        classify_junctions), provided the remaining piece is not also very
        short (e.g., < classify_max_length_m), or the 
      - the segment is > 50m long and has a single signal in the middle
        (not at the front and not near the end).
    """
    logger.info("Splitting segments with traffic signals near the start or mid-segment...")
    offset = config.get("signal_split_offset", 3.5)

    new_edges_list = []
    to_remove = []

    # Track the max existing node ID to create a new intermediate node
    max_node_id = max(edges['u'].max(), edges['v'].max())
    next_node_id = max_node_id + 1

    edges_to_process = edges.copy()

    for idx, row in edges_to_process.iterrows():
        is_oneway = str(row.get('oneway', 'False')).lower() in ['true', 'yes', '1']
        if not is_oneway:
            continue

        # Skip _link segments unless they are more than 50 meters long
        highway = str(row.get('highway', ''))
        if highway.endswith('_link') and row['length'] <= 50.0:
            continue

        sig_pos_raw = row.get('signal_positions', "[]")
        try:
            sig_positions = json.loads(sig_pos_raw)
        except:
            sig_positions = []

        if not sig_positions:
            continue

        d1 = sig_positions[0]
        d_split = None

        if len(sig_positions) >= 2:
            d_split = d1 + offset
            # Safety check: ensure split is within geometry and before the second signal
            if d_split >= row['length'] - 2.0 or d_split >= sig_positions[1]:
                d_split = None
        elif len(sig_positions) == 1 and d1 <= config.get("single_signal_split_threshold", 15.0):
            d_split = d1 + offset
            # Skip if the remaining segment would be too short (< classify_max_length_m)
            if d_split >= row['length'] - config.get("classify_max_length_m", 15.0):
                d_split = None
        elif len(sig_positions) == 1 and row['length'] > 50.0 and 0.20 <= (d1 / row['length']) <= 0.80:
            d_split = d1 + offset
            # Skip if the remaining segment would be too short (meaning signal is near the end)
            if d_split >= row['length'] - config.get("classify_max_length_m", 15.0):
                d_split = None

        if d_split is None:
            continue

        sid = str(row['segment_id'])
        logger.info(f"Splitting segment {sid} at {d_split:.1f}m (first signal at {d1:.1f}m)")

        line_a, line_b = cut_line_at_distance(row.geometry, d_split)
        if not line_a or not line_b:
            continue

        mid_node = next_node_id
        next_node_id += 1

        sid_a = sid + "_a"
        sid_b = sid + "_b"

        # Create segment A (up to split point)
        row_a = row.copy()
        row_a['segment_id'] = sid_a
        row_a['geometry'] = line_a
        row_a['length'] = line_a.length
        row_a['v'] = mid_node
        row_a['signal_positions'] = json.dumps([d1])
        row_a['signal_at_end'] = True

        # Create segment B (from split point)
        row_b = row.copy()
        row_b['segment_id'] = sid_b
        row_b['geometry'] = line_b
        row_b['length'] = line_b.length
        row_b['u'] = mid_node
        remaining_sigs = [p - d_split for p in sig_positions[1:]]
        row_b['signal_positions'] = json.dumps(remaining_sigs)
        row_b['signal_at_end'] = bool(remaining_sigs) and any(
            p >= line_b.length - 1.0 for p in remaining_sigs
        )

        new_edges_list.append(row_a)
        new_edges_list.append(row_b)
        to_remove.append(idx)

        # --- Rewire Adjacency ---
        if sid in adjacency:
            adj_ego = adjacency[sid]

            # A connects to B
            adjacency[sid_a] = adj_ego.copy()
            adjacency[sid_a]['to'] = [sid_b]
            adjacency[sid_a]['to_lengths'] = [row_b['length']]
            adjacency[sid_a]['turns_into'] = []
            adjacency[sid_a]['turns_into_lengths'] = []
            adjacency[sid_a]['merges_into'] = []
            adjacency[sid_a]['merges_into_lengths'] = []
            adjacency[sid_a]['u_turns_into'] = []
            adjacency[sid_a]['u_turns_into_lengths'] = []

            # B inherits outgoing connections from original sid
            adjacency[sid_b] = adj_ego.copy()
            adjacency[sid_b]['from'] = [sid_a]
            adjacency[sid_b]['from_lengths'] = [row_a['length']]

            # Update all other segments that pointed TO sid or came FROM sid
            for other_id, other_adj in adjacency.items():
                if other_id in [sid, sid_a, sid_b]: continue

                # Replace sid with sid_a for incoming connections
                for rel in ['to', 'turns_into', 'merges_into', 'u_turns_into']:
                    if sid in other_adj.get(rel, []):
                        idx_rel = other_adj[rel].index(sid)
                        other_adj[rel][idx_rel] = sid_a

                # Replace sid with sid_b for outgoing connections
                if sid in other_adj.get('from', []):
                    idx_from = other_adj['from'].index(sid)
                    other_adj['from'][idx_from] = sid_b

                # Replace sid in crosses / crossed_by
                if sid in other_adj.get('crosses', []):
                    idx_c = other_adj['crosses'].index(sid)
                    other_adj['crosses'][idx_c] = sid_a
                if sid in other_adj.get('crossed_by', []):
                    idx_cb = other_adj['crossed_by'].index(sid)
                    other_adj['crossed_by'][idx_cb] = sid_a
                if sid in other_adj.get('intersects_with', []):
                    idx_iw = other_adj['intersects_with'].index(sid)
                    other_adj['intersects_with'][idx_iw] = sid_a

            del adjacency[sid]

    if to_remove:
        edges = edges.drop(to_remove)
        new_edges_gdf = gpd.GeoDataFrame(new_edges_list, crs=edges.crs)
        edges = pd.concat([edges, new_edges_gdf], ignore_index=True)
        logger.info(f"Split {len(to_remove)} segments into {len(new_edges_list)} pieces.")

    return edges, adjacency


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

def get_turn_restrictions(edges_gdf, config):
    turn_restrictions_logger.info("Fetching turn restrictions from Overpass API...")
    
    # Ensure the bounding box is in Lat/Lon (EPSG:4326) for the Overpass API
    if edges_gdf.crs != "EPSG:4326":
        minx, miny, maxx, maxy = edges_gdf.to_crs("EPSG:4326").total_bounds
    else:
        minx, miny, maxx, maxy = edges_gdf.total_bounds
        
    overpass_url = "https://overpass-api.de/api/interpreter"
    buffer = config["overpass_buffer"]
    overpass_query = f"""
    [out:json][timeout:30];
    relation["type"="restriction"]({miny-buffer},{minx-buffer},{maxy+buffer},{maxx+buffer});
    out body;
    """
    
    headers = {
        'User-Agent': 'HGT-Model-Script/1.0',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers, timeout=config["overpass_timeout"])
        restrictions = []
        if response.status_code == 200:
            data = response.json()
            for element in data.get('elements', []):
                if element['type'] == 'relation':
                    tags = element.get('tags', {})
                    restriction_type = tags.get('restriction', '')
                    from_way = None
                    to_way = None
                    via_ways = []
                    via_nodes = []
                    for member in element.get('members', []):
                        role = member.get('role')
                        mtype = member.get('type')
                        if role == 'from' and mtype == 'way':
                            from_way = str(member.get('ref'))
                        elif role == 'to' and mtype == 'way':
                            to_way = str(member.get('ref'))
                        elif role == 'via' and mtype == 'way':
                            via_ways.append(str(member.get('ref')))
                        elif role == 'via' and mtype == 'node':
                            via_nodes.append(str(member.get('ref')))
                    if from_way and to_way and restriction_type:
                        restrictions.append({
                            'type': restriction_type,
                            'from': from_way,
                            'to': to_way,
                            'via_ways': via_ways,
                            'via_nodes': via_nodes
                        })
            turn_restrictions_logger.info(f"Found {len(restrictions)} turn restrictions from Overpass API.")
            return restrictions
        else:
            turn_restrictions_logger.info(f"Overpass API returned status {response.status_code}. Response: {response.text[:200]}")
    except Exception as e:
        turn_restrictions_logger.warning(f"Failed to fetch turn restrictions from API: {e}")
        
    turn_restrictions_logger.info(f"Falling back to local {config['turn_restrictions_path']}...")
    if os.path.exists(config["turn_restrictions_path"]):
        try:
            with open(config["turn_restrictions_path"], 'r') as f:
                restrictions = json.load(f)
            
            if restrictions and len(restrictions) > 0 and 'via_ways' not in restrictions[0]:
                turn_restrictions_logger.warning(f"Local {config['turn_restrictions_path']} is in an old format. Empty via_ways/via_nodes will be assumed.")
                
            for r in restrictions:
                if 'via_ways' not in r: r['via_ways'] = []
                if 'via_nodes' not in r: r['via_nodes'] = []
            turn_restrictions_logger.info(f"Successfully loaded {len(restrictions)} restrictions from local fallback.")
            return restrictions
        except Exception as e:
            turn_restrictions_logger.warning(f"Failed to load local fallback: {e}")
    else:
        turn_restrictions_logger.info("Local fallback turn_restrictions.json not found.")
        
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


def get_first_hw(x):
        return x[0] if isinstance(x, list) else x
    
    
def _dedupe_crosses_against_merges(adjacency):
    """Remove entries from crosses/crossed_by that are already in merges_with/merged_by.

    The merge relationship (sharing a successor via merges_into) is more specific
    than a generic crossing, so when both classifications apply to the same pair
    of segments we keep only the merge relationship.
    """
    for data in adjacency.values():
        all_merges = set(data.get('merges_with', [])) | set(data.get('merged_by', []))
        if not all_merges:
            continue
            
        for rel, len_key in (
            ('crosses', 'crosses_lengths'),
            ('crossed_by', 'crossed_by_lengths'),
        ):
            ids = data.get(rel, [])
            lens = data.get(len_key, [])
            kept_ids, kept_lens = [], []
            for sid, ln in zip(ids, lens):
                if sid in all_merges:
                    continue
                kept_ids.append(sid)
                kept_lens.append(ln)
            data[rel] = kept_ids
            data[len_key] = kept_lens

def _extract_mutual_intersections(adjacency):
    """Identify when an ego road segment crosses and is crossed_by the same segment.
    Move these mutual crossings into a new 'intersects_with' relationship.
    """
    for data in adjacency.values():
        if 'intersects_with' not in data:
            data['intersects_with'] = []
            data['intersects_with_lengths'] = []
            
        crosses = data.get('crosses', [])
        crossed_by = data.get('crossed_by', [])
        mutual = set(crosses).intersection(set(crossed_by))
        
        if not mutual:
            continue
            
        new_crosses = []
        new_crosses_len = []
        for c_id, c_len in zip(crosses, data.get('crosses_lengths', [])):
            if c_id in mutual:
                if c_id not in data['intersects_with']:
                    data['intersects_with'].append(c_id)
                    data['intersects_with_lengths'].append(c_len)
            else:
                new_crosses.append(c_id)
                new_crosses_len.append(c_len)
                
        data['crosses'] = new_crosses
        data['crosses_lengths'] = new_crosses_len
        
        new_crossed_by = []
        new_crossed_by_len = []
        for cb_id, cb_len in zip(crossed_by, data.get('crossed_by_lengths', [])):
            if cb_id not in mutual:
                new_crossed_by.append(cb_id)
                new_crossed_by_len.append(cb_len)
                
        data['crossed_by'] = new_crossed_by
        data['crossed_by_lengths'] = new_crossed_by_len


def build_topological_adjacency(edges_gdf, config):
    logger.info("Building topological lookups...")
    # Grouping by node IDs
    edges_by_u = edges_gdf.groupby('u')['segment_id'].apply(list).to_dict()
    edges_by_v = edges_gdf.groupby('v')['segment_id'].apply(list).to_dict()
    
    adjacency = {}
    
    
    restrictions = get_turn_restrictions(edges_gdf, config)
    
    with open(config["turn_restrictions_path"], "w") as f:
        json.dump(restrictions, f, indent=4)
    turn_restrictions_logger.info(f"Saved all extracted turn restrictions to {config['turn_restrictions_path']} for inspection.")
        
    banned_transitions = []
    only_transitions = []

    # Immediate-successor restrictions: from-way directly connected to to-way (no via-ways).
    # Via-way restrictions are handled later during junction propagation.
    for r in restrictions:
        if r.get('via_ways'):
            continue
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
        name_val = row['name'] if 'name' in row else ''
        segment_data[sid] = {
            'u': row['u'],
            'v': row['v'],
            'osmid': str(row['osmid']),
            'highway': str(row['highway']),
            'name': get_first_hw(name_val),
            'length': float(row['length']),
            'exit_heading': get_smoothed_heading(geom, reverse=False, look_dist=config["heading_look_dist"]),
            'entry_heading': get_smoothed_heading(geom, reverse=True, look_dist=config["heading_look_dist"])
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
            
            if -config["angle_straight_thresh"] <= delta_theta <= config["angle_straight_thresh"]: direction = 'straight'
            elif config["angle_straight_thresh"] < delta_theta <= config["angle_left_right_limit_basic"]: direction = 'left'
            elif -config["angle_left_right_limit_basic"] <= delta_theta < -config["angle_straight_thresh"]: direction = 'right'
            elif config["angle_u_turn_thresh"] <= abs(delta_theta) <= 180: direction = 'u_turn'
            else: continue
            
            # Apply U-Turn Filters (Undivided Road, Mid-Block, Highway Classification)
            if direction == 'u_turn':
                is_same_way = any(e_id in succ_osmids for e_id in ego_osmids)
                hw_type = segment_data[ego_id]['highway']
                banned_hw = config["banned_u_turn_hw_types"]
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
                    turn_restrictions_logger.info(f"Ego [{ego_id}]: Mandatory {direction} transition to segment {sid_str} confirmed ({matched_rtypes[0]}).")
                else:
                    is_banned = True
                    rtypes = list(set(only_targets_succ.values()))
                    turn_restrictions_logger.info(f"Ego [{ego_id}] ({ego_osmids}) : Connection to segment {sid_str} filtered out (Only restriction {rtypes} points to segment(s) {target_sids}).")
                    
            if not is_banned and not is_only:
                for e_id in ego_osmids:
                    for s_id in succ_osmids:
                        for (f_id, t_id, rtype) in banned_transitions:
                            if f_id == e_id and t_id == s_id:
                                if direction in rtype:
                                    is_banned = True
                                    turn_restrictions_logger.info(f"Ego [{ego_id}] ({e_id}): Banned connection to segment {sid_str} removed by spatial filter ({rtype} explicitly matched {direction}).")
                                    
            if is_banned:
                banned_successors.append(sid_str)
            else:
                valid_successors.append(sid_str)
                if direction == 'u_turn':
                    u_turn_successors.append(sid_str)
                    junctions_logger.info(f"Ego [{ego_id}]: Valid U-Turn to successor segment {sid_str} detected.")
                
                if is_only:
                    only_successors.append(sid_str)
                    if 'straight' in matched_rtypes[0]:
                        best_succ = sid_str
                        min_delta_succ = -1.0
                else:
                    if min_delta_succ == -1.0: continue
                    
                    is_same_way = any(e_id in succ_osmids for e_id in ego_osmids)
                    threshold = config["angle_same_way_thresh"] if is_same_way else config["angle_diff_way_thresh"]
                    
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
        non_link_preds = [p for p in potential_predecessors if not segment_data[str(p)]['highway'].endswith('_link')]
        link_preds = [p for p in potential_predecessors if segment_data[str(p)]['highway'].endswith('_link')]
                
        def evaluate_predecessors(preds_list):
            b_pred = None
            m_delta = float('inf')
            for pred_id in preds_list:
                sid_str = str(pred_id)
                if sid_str == ego_id: continue
                
                pred_osmids = parse_osmids(segment_data[sid_str]['osmid'])
                pred_heading = segment_data[sid_str]['exit_heading']
                delta_theta = (data['entry_heading'] - pred_heading + 180) % 360 - 180
                
                if -config["angle_straight_thresh"] <= delta_theta <= config["angle_straight_thresh"]: direction = 'straight'
                elif config["angle_straight_thresh"] < delta_theta <= config["angle_left_right_limit_basic"]: direction = 'left'
                elif -config["angle_left_right_limit_basic"] <= delta_theta < -config["angle_straight_thresh"]: direction = 'right'
                elif config["angle_u_turn_thresh"] <= abs(delta_theta) <= 180: direction = 'u_turn'
                else: continue
                
                if direction != 'u_turn' and abs(delta_theta) > config["angle_pred_limit"]:
                    continue
                
                # Apply U-Turn Filters for Predecessors
                if direction == 'u_turn':
                    is_same_way = any(p_id in ego_osmids for p_id in pred_osmids)
                    hw_type = segment_data[ego_id]['highway']
                    banned_hw = config["banned_u_turn_hw_types"]
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
                        turn_restrictions_logger.info(f"Ego [{ego_id}]: Mandatory transition from predecessor segment {sid_str} confirmed ({matched_rtypes[0]}).")
                    else:
                        is_banned = True
                        rtypes = list(set(targets.values()))
                        turn_restrictions_logger.info(f"Ego [{ego_id}] ({ego_osmids}): Predecessor segment {sid_str} filtered out (Only restriction {rtypes} points to segment(s) {target_sids_pred}).")
                        
                if not is_banned and not is_only:
                    for p_id in pred_osmids:
                        for e_id in ego_osmids:
                            for (f_id, t_id, rtype) in banned_transitions:
                                if f_id == p_id and t_id == e_id:
                                    if direction in rtype:
                                        is_banned = True
                                        turn_restrictions_logger.info(f"Ego [{ego_id}] ({ego_osmids}): Banned connection from predecessor segment {sid_str} removed by spatial filter ({rtype} explicitly matched {direction}).")
                                        
                if is_banned:
                    continue
                
                is_same_way = any(p_id in ego_osmids for p_id in pred_osmids)
                threshold = config["angle_same_way_thresh"] if is_same_way else config["angle_diff_way_thresh"]
                
                if is_only:
                    if 'straight' in matched_rtypes[0]:
                        m_delta = -1.0
                        b_pred = sid_str
                        break
                    else:
                        continue
                
                if abs(delta_theta) <= threshold and abs(delta_theta) < m_delta:
                    m_delta = abs(delta_theta)
                    b_pred = sid_str
            return b_pred, m_delta
            
        best_pred, min_delta_pred = evaluate_predecessors(non_link_preds)
        if not best_pred:
            best_pred, min_delta_pred = evaluate_predecessors(link_preds)
                
        best_immediate[ego_id] = {
            'successor': best_succ, 
            'predecessor': best_pred,
            'banned_successors': banned_successors,
            'only_successors': only_successors,
            'valid_successors': valid_successors,
            'u_turn_successors': u_turn_successors
        }

    logger.info(f"Chaining segments to reach {config['target_dist']}m...")
    for ego_id in segment_data.keys():
        # Build Successor Chain
        succ_chain = []
        merging_into = []
        curr_dist = 0.0
        curr_id = ego_id
        while curr_dist < config["target_dist"]:
            next_id = best_immediate[curr_id]['successor']
            if not next_id or next_id in succ_chain or next_id == ego_id:
                break
                
            # Merge check: Only consider it a continuation if we are the preferred predecessor
            right_of_way_pred = best_immediate[next_id]['predecessor']
            if right_of_way_pred != curr_id:
                if curr_id == ego_id:
                    merging_into.append(next_id)
                    junctions_logger.info(f"Ego [{ego_id}]: Merging into segment {next_id} detected.")
                break # We yield right of way, so our topological successor chain ends here
                
            succ_chain.append(next_id)
            curr_dist += segment_data[next_id]['length']
            curr_id = next_id
            
        # Build Predecessor Chain
        pred_chain = []
        curr_dist = 0.0
        curr_id = ego_id
        while curr_dist < config["target_dist"]:
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
            filtered_intersects_out = [str(sid) for sid in only_successors if str(sid) != immediate_succ and str(sid) not in opposite_direction and str(sid) not in u_turns and str(sid) not in merging_into]
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
            'intersects_with': [],
            'banned_successors': banned_successors,
            'only_successors': only_successors,
            'opposite_direction': opposite_direction,
            'merges_with': [],
            'merged_by': [],
            'to_lengths': [segment_data[sid]['length'] for sid in succ_chain],
            'from_lengths': [segment_data[sid]['length'] for sid in pred_chain],
            'merges_into_lengths': [segment_data[sid]['length'] for sid in merging_into],
            'crossed_by_lengths': [segment_data[sid]['length'] for sid in filtered_intersects_in],
            'turns_into_lengths': [segment_data[sid]['length'] for sid in filtered_intersects_out],
            'u_turns_into_lengths': [segment_data[sid]['length'] for sid in u_turns],
            'crosses_lengths': [segment_data[sid]['length'] for sid in crosses],
            'intersects_with_lengths': [],
            'opposite_direction_lengths': [segment_data[sid]['length'] for sid in opposite_direction],
            'merges_with_lengths': [],
            'merged_by_lengths': [],
            'to_junction_chain': [],
            'to_junction_chain_lengths': [],
            'from_junction_chain': [],
            'from_junction_chain_lengths': []
        }
        
    logger.info("Applying post-processing transition rules for junction links...")
    for ego_id, data in adjacency.items():
        turns_into = data.get('turns_into', [])
        for L in turns_into:
            if segment_data[L]['highway'].endswith('_link'):
                M_list = adjacency.get(L, {}).get('merges_into', []) + adjacency.get(L, {}).get('turns_into', [])
                if not M_list: continue
                
                M_names = [segment_data[m].get('name') for m in M_list if segment_data[m].get('name')]
                if not M_names: continue
                M_headings = [segment_data[m]['entry_heading'] for m in M_list]
                
                succs = data.get('to', [])[:3]
                for S_i in succs:
                    if S_i not in adjacency: continue
                    S_i_turns = adjacency[S_i].get('turns_into', [])
                    
                    T_to_remove = []
                    for T in S_i_turns:
                        T_name = segment_data[T].get('name')
                        if T_name in M_names:
                            T_heading = segment_data[T]['entry_heading']
                            # Check if direction is roughly the same as any M (i.e. not the opposite side of the road)
                            is_same_dir = any(abs((T_heading - m_h + 180) % 360 - 180) < config["angle_parallel_thresh"] for m_h in M_headings)
                            if is_same_dir:
                                T_to_remove.append(T)
                                
                    for T in T_to_remove:
                        if T in adjacency[S_i]['turns_into']:
                            idx = adjacency[S_i]['turns_into'].index(T)
                            adjacency[S_i]['turns_into'].pop(idx)
                            adjacency[S_i]['turns_into_lengths'].pop(idx)
                            
                            # Clean up crosses and crossed_by lists that were built assuming S_i turns into T
                            crosses_to_remove = [
                                b_str for b_str in adjacency[S_i].get('crosses', [])
                                if best_immediate.get(b_str, {}).get('successor') == T
                            ]
                            for b_str in crosses_to_remove:
                                if b_str in adjacency[S_i]['crosses']:
                                    c_idx = adjacency[S_i]['crosses'].index(b_str)
                                    adjacency[S_i]['crosses'].pop(c_idx)
                                    adjacency[S_i]['crosses_lengths'].pop(c_idx)
                                    
                                if b_str in adjacency and S_i in adjacency[b_str].get('crossed_by', []):
                                    cb_idx = adjacency[b_str]['crossed_by'].index(S_i)
                                    adjacency[b_str]['crossed_by'].pop(cb_idx)
                                    adjacency[b_str]['crossed_by_lengths'].pop(cb_idx)
                                    
                            junctions_logger.info(f"Post-processing: Removed redundant turn {T} from {S_i} (parallels _link {L} merge/turn). Updated cross relations.")

    logger.info("Calculating merges_with and merged_by relations...")
    for ego_id, data in adjacency.items():
        data['merges_with'] = []
        data['merges_with_lengths'] = []
        data['merged_by'] = []
        data['merged_by_lengths'] = []

    for ego_id, data in adjacency.items():
        for A in data.get('merges_into', []):
            if A in adjacency:
                from_segs = adjacency[A].get('from', [])
                if from_segs:
                    P = from_segs[0]
                    if P not in adjacency[ego_id]['merges_with']:
                        adjacency[ego_id]['merges_with'].append(P)
                        adjacency[ego_id]['merges_with_lengths'].append(segment_data[P]['length'])
                    if ego_id not in adjacency[P]['merged_by']:
                        adjacency[P]['merged_by'].append(ego_id)
                        adjacency[P]['merged_by_lengths'].append(segment_data[ego_id]['length'])

    # Ensure merges_with/merged_by and crosses/crossed_by are mutually exclusive:
    # a pair with a shared-successor merge relationship should not also be marked
    # as crossing. The merge relationship is the more specific topological fact.
    _dedupe_crosses_against_merges(adjacency)
    _extract_mutual_intersections(adjacency)

    return adjacency, restrictions

def has_crossing_edges_at_connection(geom_J, geom_target, is_succ, gdf, s_id, target_id, deleted, tree, tree_geoms, tree_ids):
    """
    Returns True if any other edge's geometry crosses the short corridor formed
    by the junction link (geom_J) and target segment (geom_target) at their
    connection point. Used to block geometrically unsafe merges.

    Parameters
    ----------
    geom_J      : shapely geometry of the junction link
    geom_target : shapely geometry of the target segment
    is_succ     : True if target is a successor (J -> target), False if predecessor
    gdf         : full GeoDataFrame indexed by segment_id
    s_id        : segment_id of the junction link
    target_id   : segment_id of the target segment
    deleted     : set of already-deleted segment ids
    tree        : STRtree built from all gdf geometries (built once outside the loop)
    tree_geoms  : list of geometries parallel to the STRtree
    tree_ids    : list of segment_ids parallel to the STRtree
    """
    BUFFER_RADIUS = 3.0   # metres around connection point for candidate search
    CORRIDOR_HALF_WIDTH = 1.0  # metres buffer around the short corridor LineString

    # --- connection point and short corridor ---
    coords_J = list(geom_J.coords)
    coords_T = list(geom_target.coords)

    if is_succ:
        # Junction link flows INTO target: last point of J == first point of T
        conn_pt = Point(coords_J[-1])
        # Take the last 2 coords of J and first 2 coords of T as the corridor
        corridor_coords = coords_J[-2:] + coords_T[:2]
    else:
        # Target flows INTO junction link: last point of T == first point of J
        conn_pt = Point(coords_J[0])
        # Take last 2 coords of T and first 2 coords of J as the corridor
        corridor_coords = coords_T[-2:] + coords_J[:2]

    # Deduplicate consecutive duplicate coordinates (can happen at shared nodes)
    deduped = [corridor_coords[0]]
    for c in corridor_coords[1:]:
        if c != deduped[-1]:
            deduped.append(c)

    if len(deduped) < 2:
        return False

    corridor_line = LineString(deduped)
    corridor_buf = corridor_line.buffer(CORRIDOR_HALF_WIDTH)

    # --- spatial query ---
    # Edges that share an endpoint at the connection node (e.g. opposite-
    # direction pair of a two-way road) are topologically connected, not
    # crossing through.  Collect the node id so we can skip them.
    if is_succ:
        conn_node = gdf.at[s_id, 'v']
    else:
        conn_node = gdf.at[s_id, 'u']

    search_buf = conn_pt.buffer(BUFFER_RADIUS)
    candidate_indices = tree.query(search_buf)

    for idx in candidate_indices:
        cand_id = tree_ids[idx]
        if cand_id == s_id or cand_id == target_id:
            continue
        if cand_id in deleted:
            continue
        # Skip edges that share the connection node — they connect to it,
        # not pass through it (typical for opposite-direction pairs).
        if cand_id in gdf.index:
            if gdf.at[cand_id, 'u'] == conn_node or gdf.at[cand_id, 'v'] == conn_node:
                continue
        cand_geom = tree_geoms[idx]
        if cand_geom.crosses(corridor_buf):
            return True

    return False


def merge_short_segments(gdf, adj, config):
    """
    Two merge modes operate on junction links that are not semantic/manual:

    Standalone mode (is_shape_junction=False): the junction link is absorbed
    into its non-junction neighbor. The neighbor keeps its id and inherits
    the extended geometry; the junction link is deleted.

    Shape mode (is_shape_junction=True, length <= classify_max_length_m):
    the shape-junction edge is merged with a connected non-junction road
    segment. Here the shape edge keeps its id so the shape's ring stays
    intact, but its junction flags are cleared and it absorbs the road
    segment's attributes + geometry; the road segment is deleted.
    """
    logger.info("Merging standalone and short shape junction links into their neighbors...")
    gdf = gdf.copy()
    gdf.set_index('segment_id', inplace=True, drop=False)

    # Track segments produced by a junction-pair merge that exceeded max_len and
    # were unflagged as junction links. build_controllers uses this to avoid
    # reassigning them back to junction status.
    gdf['was_merged'] = 'False'

    junction_sids = set(gdf[gdf['is_internal_junction'].astype(str).str.lower() == 'true'].index.astype(str))
    deleted = set()
    standalone_count = 0
    shape_count = 0

    # Build STRtree once for geometrical crossing checks (tree is static;
    # deleted edges are filtered out at query time).
    _tree_ids = list(gdf.index.astype(str))
    _tree_geoms = list(gdf['geometry'])
    _strtree = STRtree(_tree_geoms)

    max_len = config["classify_max_length_m"]
    look_dist = config["heading_look_dist"]

    # Build undirected graph of junction edges to detect fully closed shapes
    junc_adj_undir = {}
    for sid in junction_sids:
        u = gdf.at[sid, 'u']
        v = gdf.at[sid, 'v']
        junc_adj_undir.setdefault(u, []).append((v, sid))
        junc_adj_undir.setdefault(v, []).append((u, sid))
        
    def in_closed_shape(edge_id, start_u, start_v):
        if start_u == start_v:
            return True
            
        visited = {start_v}
        queue = [start_v]
        forbidden_endpoints = frozenset((start_u, start_v))
        
        while queue:
            curr = queue.pop(0)
            for nxt_node, nxt_sid in junc_adj_undir.get(curr, []):
                if nxt_sid == edge_id: continue
                if nxt_sid in deleted: continue
                if str(gdf.at[nxt_sid, 'is_internal_junction']).lower() != 'true': continue
                
                # Prevent trivially hopping back and forth on the exact same endpoint pair
                if frozenset((curr, nxt_node)) == forbidden_endpoints:
                    continue
                    
                if nxt_node == start_u:
                    return True
                    
                if nxt_node not in visited:
                    visited.add(nxt_node)
                    queue.append(nxt_node)
        return False

    for s_id in list(junction_sids):
        if s_id in deleted: continue
        if s_id not in gdf.index: continue

        is_shape = str(gdf.at[s_id, 'is_shape_junction']).lower() == 'true'
        is_semantic = str(gdf.at[s_id, 'is_semantic_junction']).lower() == 'true'
        is_manual = str(gdf.at[s_id, 'is_manual_junction']).lower() == 'true' if 'is_manual_junction' in gdf.columns else False

        if is_semantic or is_manual:
            continue

        shape_mode = is_shape
        if shape_mode:
            if float(gdf.at[s_id, 'length']) > max_len:
                continue

            # Only merge if the shape-junction edge is NOT part of a fully closed shape
            u_node = gdf.at[s_id, 'u']
            v_node = gdf.at[s_id, 'v']
            if in_closed_shape(s_id, u_node, v_node):
                continue

        s_adj = adj.get(str(s_id))
        if not s_adj: continue

        if s_adj.get('merged_by'):
            junctions_logger.info(f"Skipping merge for {s_id}: it is 'merged_by' another segment.")
            continue

        if shape_mode:
            # A shape-junction edge with branching turns or incoming crossings
            # represents real topological complexity — collapsing it into a
            # single neighbor would silently erase those relationships.
            if s_adj.get('turns_into') or s_adj.get('crossed_by'):
                junctions_logger.info(
                    f"Skipping shape-junction merge for {s_id}: has "
                    f"turns_into={s_adj.get('turns_into')} crossed_by={s_adj.get('crossed_by')}."
                )
                continue

        if not shape_mode:
            # Standalone mode: skip if any connected neighbor is itself a junction.
            connected_sids = set()
            for rel in ['turns_into', 'merges_into', 'u_turns_into', 'crosses', 'crossed_by', 'intersects_with']:
                connected_sids.update(s_adj.get(rel, []))
            if s_adj.get('to'):
                connected_sids.add(str(s_adj.get('to')[0]))
            if s_adj.get('from'):
                connected_sids.add(str(s_adj.get('from')[0]))
            if connected_sids.intersection(junction_sids):
                continue

        S_list = s_adj.get('to', [])
        P_list = s_adj.get('from', [])
        S = str(S_list[0]) if S_list else None
        P = str(P_list[0]) if P_list else None

        if shape_mode:
            # Candidate must be a non-junction road segment (is_shape_junction=False
            # implied, and no other junction flag either).
            S_valid = bool(
                S and S in gdf.index and S not in junction_sids
                and S != s_id and S not in deleted
            )
            P_valid = bool(
                P and P in gdf.index and P not in junction_sids
                and P != s_id and P not in deleted
            )
        else:
            S_valid = False
            if S and P:
                S_adj = adj.get(S, {})
                S_from = S_adj.get('from', [])
                S_from_0 = str(S_from[0]) if S_from else None
                if S_from_0 == P and not s_adj.get('crossed_by') and not s_adj.get('turns_into'):
                    S_valid = True

            P_valid = False
            if P and S:
                P_adj = adj.get(P, {})
                P_to = P_adj.get('to', [])
                P_to_0 = str(P_to[0]) if P_to else None
                if P_to_0 == S and not P_adj.get('turns_into') and not P_adj.get('crossed_by'):
                    P_valid = True

        if not S_valid and not P_valid:
            continue

        geom_J = gdf.loc[s_id, 'geometry']
        delta_S = float('inf')
        delta_P = float('inf')

        if S_valid:
            geom_S = gdf.loc[S, 'geometry']
            delta_S = abs((get_smoothed_heading(geom_S, reverse=True, look_dist=look_dist) - get_smoothed_heading(geom_J, reverse=False, look_dist=look_dist) + 180) % 360 - 180)
        if P_valid:
            geom_P = gdf.loc[P, 'geometry']
            delta_P = abs((get_smoothed_heading(geom_J, reverse=True, look_dist=look_dist) - get_smoothed_heading(geom_P, reverse=False, look_dist=look_dist) + 180) % 360 - 180)

        target_id = None
        is_succ = True

        if S_valid and P_valid:
            if delta_S <= delta_P:
                target_id = S; is_succ = True
            else:
                target_id = P; is_succ = False
        elif S_valid:
            target_id = S; is_succ = True
        elif P_valid:
            target_id = P; is_succ = False

        if not target_id or target_id == s_id or target_id not in gdf.index or target_id in deleted:
            continue

        # Geometrical crossing check: block the merge if another edge cuts
        # through the corridor where the junction link meets the target segment.
        if has_crossing_edges_at_connection(
            gdf.loc[s_id, 'geometry'], gdf.loc[target_id, 'geometry'],
            is_succ, gdf, s_id, target_id, deleted,
            _strtree, _tree_geoms, _tree_ids
        ):
            if shape_mode:
                junctions_logger.info(
                    f"Skipping shape-junction merge for {s_id} -> {target_id}: "
                    f"crossing edge detected at connection point."
                )
            else:
                logger.info(
                    f"Skipping standalone junction merge for {s_id} -> {target_id}: "
                    f"crossing edge detected at connection point."
                )
            continue

        s_row, t_row = gdf.loc[s_id], gdf.loc[target_id]
        s_coords = list(s_row['geometry'].coords)
        t_coords = list(t_row['geometry'].coords)

        if is_succ:
            new_coords = s_coords + t_coords[1:] if s_coords[-1] == t_coords[0] else s_coords + t_coords
        else:
            new_coords = t_coords + s_coords[1:] if t_coords[-1] == s_coords[0] else t_coords + s_coords

        merged_geom = LineString(new_coords)

        gdf.at[target_id, 'geometry'] = merged_geom
        gdf.at[target_id, 'length'] = merged_geom.length
        if is_succ:
            gdf.at[target_id, 'u'] = s_row['u']
        else:
            gdf.at[target_id, 'v'] = s_row['v']

        deleted.add(s_id)
        if str(s_id) in adj:
            del adj[str(s_id)]

        if shape_mode:
            shape_count += 1
            junctions_logger.info(
                f"Merged shape-junction {s_id} with {'successor' if is_succ else 'predecessor'} "
                f"non-junction {target_id}; absorbed shape into {target_id}."
            )
        else:
            standalone_count += 1
            logger.info(
                f"Merged junction link {s_id} into {'successor' if is_succ else 'predecessor'} "
                f"segment {target_id}. Updated geometry and nodes."
            )

    logger.info(f"Absorbed {standalone_count} standalone junction link(s); "
                f"merged {shape_count} shape-junction link(s) with adjacent roads.")
    gdf = gdf[~gdf.index.isin(deleted)].copy()
    gdf.reset_index(drop=True, inplace=True)
    return gdf

def merge_junction_pairs(gdf, adj, config):
    """
    Greedy pairwise merge of adjacent junction links whose only relationship is
    a simple to/from continuation. A qualifies as the "from" side only if its
    crossed_by, crosses, turns_into, merges_into, and u_turns_into are all empty.
    B must be a junction link that lists A in its 'from'. B's attributes are
    retained; if oneway differs, lanes come from the two-way side. A merged
    segment longer than classify_max_length_m is unflagged as a junction.
    """
    logger.info("Merging junction link pairs with simple to/from chains...")
    gdf = gdf.copy()
    gdf.set_index('segment_id', inplace=True, drop=False)
    max_len = config["classify_max_length_m"]
    total = 0

    while True:
        junction_sids = set(
            gdf[gdf['is_internal_junction'].astype(str).str.lower() == 'true']
            .index.astype(str)
        )
        found = None
        for a_id in sorted(junction_sids):
            a_adj = adj.get(a_id)
            if not a_adj:
                continue
            if (a_adj.get('crossed_by') or a_adj.get('crosses')
                    or a_adj.get('intersects_with')
                    or a_adj.get('turns_into') or a_adj.get('merges_into')
                    or a_adj.get('u_turns_into')):
                continue
            a_to = a_adj.get('to', [])
            if not a_to:
                continue
            b_id = str(a_to[0])
            if b_id == a_id or b_id not in junction_sids:
                continue
            b_adj = adj.get(b_id)
            if not b_adj or a_id not in b_adj.get('from', []):
                continue
            # Restrict to pairings between a shape-junction and a standalone
            # junction (exactly one side has is_shape_junction=True).
            a_shape = str(gdf.at[a_id, 'is_shape_junction']).lower() == 'true'
            b_shape = str(gdf.at[b_id, 'is_shape_junction']).lower() == 'true'
            if a_shape == b_shape:
                continue
            found = (a_id, b_id)
            break

        if not found:
            break

        a_id, b_id = found
        a_row = gdf.loc[a_id]
        b_row = gdf.loc[b_id]

        a_coords = list(a_row['geometry'].coords)
        b_coords = list(b_row['geometry'].coords)
        new_coords = a_coords + b_coords[1:] if a_coords[-1] == b_coords[0] else a_coords + b_coords
        merged_geom = LineString(new_coords)
        merged_len = merged_geom.length

        gdf.at[b_id, 'geometry'] = merged_geom
        gdf.at[b_id, 'length'] = merged_len
        gdf.at[b_id, 'u'] = a_row['u']

        a_oneway = str(a_row.get('oneway', 'False')).lower() in ['true', 'yes', '1']
        b_oneway = str(b_row.get('oneway', 'False')).lower() in ['true', 'yes', '1']
        if a_oneway != b_oneway and not a_oneway:
            gdf.at[b_id, 'lanes'] = a_row.get('lanes')

        if merged_len > max_len:
            gdf.at[b_id, 'is_internal_junction'] = 'False'
            if 'is_shape_junction' in gdf.columns:
                gdf.at[b_id, 'is_shape_junction'] = 'False'
            gdf.at[b_id, 'was_merged'] = 'True'
            junctions_logger.info(
                f"Merged junction pair {a_id}+{b_id} -> {b_id} "
                f"(length {merged_len:.2f}m > {max_len}m; unflagged as junction)."
            )
        else:
            junctions_logger.info(
                f"Merged junction pair {a_id}+{b_id} -> {b_id} (length {merged_len:.2f}m)."
            )

        # Minimal adjacency fix-up so subsequent iterations can chain further pairs.
        b_from = [x for x in adj[b_id].get('from', []) if x != a_id]
        for p in adj[a_id].get('from', []):
            if p not in b_from:
                b_from.append(p)
        adj[b_id]['from'] = b_from
        adj[b_id]['from_lengths'] = [
            float(gdf.at[p, 'length']) if p in gdf.index else 0.0 for p in b_from
        ]
        del adj[a_id]
        gdf = gdf.drop(index=a_id)
        total += 1

    logger.info(f"Merged {total} junction link pair(s).")
    gdf.reset_index(drop=True, inplace=True)
    return gdf

def unflag_long_link_junctions(edges_gdf, config):
    """
    Unflags long *_link segments as junction links if they host a single traffic signal.
    This prevents them from being bypassed during junction propagation, allowing vehicles
    to queue on them appropriately.
    """
    max_length_m = config.get("classify_max_length_m", 15.0)
    unflagged_count = 0
    
    for idx, row in edges_gdf.iterrows():
        # Must be classified as an internal junction
        if str(row.get('is_internal_junction', 'False')).lower() != 'true':
            continue
            
        # Must be a _link road
        highway = str(row.get('highway', ''))
        if not highway.endswith('_link'):
            continue
            
        # Must be longer than the threshold
        if float(row.get('length', 0.0)) <= max_length_m:
            continue
            
        # Must have exactly one signal
        sig_pos_raw = row.get('signal_positions', "[]")
        try:
            sig_positions = json.loads(sig_pos_raw)
        except:
            sig_positions = []
            
        if len(sig_positions) == 1:
            edges_gdf.at[idx, 'is_internal_junction'] = 'False'
            edges_gdf.at[idx, 'is_shape_junction'] = 'False'
            edges_gdf.at[idx, 'is_long_shape_boundary'] = 'False'
            unflagged_count += 1
            junctions_logger.info(
                f"Unflagged long _link segment {row['segment_id']} (length {row['length']:.1f}m) "
                f"as a junction link because it hosts a single traffic signal."
            )
            
    if unflagged_count > 0:
        junctions_logger.info(f"Unflagged {unflagged_count} long signalized _link segments in total.")
        
    return edges_gdf

def classify_junctions(edges_gdf, config):
    from shapely.ops import polygonize
    max_area_sqm = config["classify_max_area_sqm"]
    max_length_m = config["classify_max_length_m"]
    junctions_logger.info(f"Classifying junctions (max area: {max_area_sqm} sqm, max length: {max_length_m}m)...")

    # Deduplicate by undirected endpoint pair: two-way roads produce two directed
    # edges with overlapping reversed geometry, which blocks polygonize from closing
    # shapes that include a two-way segment.
    seen_pairs = set()
    unique_geoms = []
    for u, v, geom in zip(edges_gdf['u'], edges_gdf['v'], edges_gdf.geometry):
        key = frozenset((u, v))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_geoms.append(geom)

    # Use spatial properties of the lines to find closed shapes
    polygons = list(polygonize(unique_geoms))
    
    # Filter polygons by area threshold
    junction_polys = [poly for poly in polygons if poly.area <= max_area_sqm]
    
    junction_edges_idx = set()
    
    # Condition 1: Short edges
    short_edges = set(edges_gdf[edges_gdf['length'] <= max_length_m].index)
    junction_edges_idx.update(short_edges)
    
    # Condition 2: Edges forming small shapes
    shape_edges = set()
    long_boundary_edges = set()   # on the boundary but length > classify_shape_max_length_m
    if junction_polys:
        sindex = edges_gdf.sindex
        for poly in junction_polys:
            bound = poly.boundary
            possible_matches_index = list(sindex.intersection(bound.bounds))
            poly_edge_indices = []
            for idx in possible_matches_index:
                row = edges_gdf.iloc[idx]
                geom = row.geometry
                # Only select edges that make up the boundary (intersection length > 0),
                # ignoring external edges that just touch the boundary at a vertex.
                if geom.intersection(bound).length > config["classify_min_intersection_length"]:
                    if row['length'] <= config["classify_shape_max_length_m"]:
                        poly_edge_indices.append(idx)
                    else:
                        # Edge is on the polygon boundary but too long to be classified
                        # as a junction link here. Track it so build_controllers can
                        # reassign it if the polygon becomes part of a junction group.
                        long_boundary_edges.add(idx)
            shape_edges.update(poly_edge_indices)
            junction_edges_idx.update(poly_edge_indices)

    # Condition 3: Semantic tags (OSM junction=* tag)
    if 'junction' in edges_gdf.columns:
        semantic_edges = set(edges_gdf[~edges_gdf['junction'].isin(['none', 'nan', 'no', '', 'null', '<na>'])].index)
        junction_edges_idx.update(semantic_edges)
    else:
        semantic_edges = set()

    # Condition 4: Manual override from segments_junctions.json
    manual_junctions = []
    if os.path.exists(config.get("segments_junctions_path", "")):
        try:
            with open(config["segments_junctions_path"], 'r') as f:
                manual_junctions = [str(x) for x in json.load(f)]
            junctions_logger.info(f"Loaded {len(manual_junctions)} manual junction segments from {config['segments_junctions_path']}.")
        except Exception as e:
            junctions_logger.warning(f"Failed to load manual junctions: {e}")

    manual_edges = set()
    if manual_junctions:
        manual_edges = set(edges_gdf[edges_gdf['segment_id'].astype(str).isin(manual_junctions)].index)
        junction_edges_idx.update(manual_edges)

    edges_gdf['is_internal_junction'] = 'False'
    edges_gdf.loc[list(junction_edges_idx), 'is_internal_junction'] = 'True'

    edges_gdf['is_shape_junction'] = 'False'
    edges_gdf.loc[list(shape_edges), 'is_shape_junction'] = 'True'

    # Edges that lie on a junction polygon boundary but whose length exceeds
    # classify_shape_max_length_m are tracked here so build_controllers can
    # reassign them as junction links when the polygon forms a junction group.
    edges_gdf['is_long_shape_boundary'] = 'False'
    # Exclude any that were also classified as junction links through other conditions
    long_boundary_only = long_boundary_edges - junction_edges_idx
    edges_gdf.loc[list(long_boundary_only), 'is_long_shape_boundary'] = 'True'

    edges_gdf['is_semantic_junction'] = 'False'
    edges_gdf.loc[list(semantic_edges), 'is_semantic_junction'] = 'True'

    edges_gdf['is_manual_junction'] = 'False'
    edges_gdf.loc[list(manual_edges), 'is_manual_junction'] = 'True'

    num_total = len(junction_edges_idx)
    num_short = len(short_edges)
    num_semantic = len(semantic_edges)
    num_manual = len(manual_edges)
    num_shape_only = num_total - len(short_edges.union(semantic_edges).union(manual_edges))
    
    junctions_logger.info(f"Classified {num_total} edges as junctions ({num_semantic} from semantic tags, {num_manual} from manual overrides, {num_shape_only} purely from shapes, and {num_short} from length).")
    
    return edges_gdf

def propagate_junction_topology(adjacency, edges_gdf, restrictions, config):
    """
    Step 7 of the pipeline.

    After classify_junctions has flagged edges belonging to the internal
    geometry of a junction, rewire the topology so that external road
    segments connect directly to the segments on the far side of the
    junction, bypassing the chain of internal junction links.

    Rules
    -----
    * Junction links may stack ('to' -> 'to', 'turns_into' -> 'to', or
      'merges_into' -> 'to', even forming circular roundabouts), so the
      chain is searched consecutively until a non-junction segment is
      reached.
    * If every hop in the chain is a 'to' continuation, the ego is
      connected 'to' the final external segment (preserving through-
      movement). Otherwise the relationship is classified by the
      geometric heading delta between ego and the external target:
      straight + merge-hop -> merges_into, left/right -> turns_into,
      reversed -> u_turns_into.
    * Way-based turn restrictions (OSM relations with one or more via
      ways) are honored: a chain whose traversed ways match the via list
      of a 'no_*' restriction is dropped, and an 'only_*' restriction
      forces the ego to keep only chains that land on the mandated
      target.
    * crosses and crossed_by are recomputed for every affected
      non-junction segment using the union of the junction footprint
      nodes as the conceptual intersection.
    """
    junctions_logger.info("Propagating topology through classified junction links...")

    junction_set = set(
        edges_gdf.loc[edges_gdf['is_internal_junction'] == 'True', 'segment_id']
        .astype(str)
        .tolist()
    )
    if not junction_set:
        junctions_logger.info("No junction links classified; skipping propagation.")
        return adjacency

    # Rebuild lightweight per-segment metadata. segment_ids may have been
    # reissued by merge_short_segments, so we always derive from edges_gdf.
    segment_data = {}
    for _, row in edges_gdf.iterrows():
        sid = str(row['segment_id'])
        geom = row['geometry']
        name_val = row['name'] if 'name' in row else ''
        segment_data[sid] = {
            'u': row['u'],
            'v': row['v'],
            'osmid': str(row['osmid']),
            'highway': str(row['highway']),
            'name': get_first_hw(name_val),
            'length': float(row['length']),
            'exit_heading': get_smoothed_heading(geom, reverse=False, look_dist=config["heading_look_dist"]),
            'entry_heading': get_smoothed_heading(geom, reverse=True, look_dist=config["heading_look_dist"]),
        }

    edges_by_u = edges_gdf.groupby('u')['segment_id'].apply(list).to_dict()
    edges_by_v = edges_gdf.groupby('v')['segment_id'].apply(list).to_dict()

    # Split ALL restrictions into banned / mandatory tables for junction propagation.
    banned_rels = []  # (from_way, [via_ways], to_way, rtype)
    only_rels = []
    for r in restrictions:
        vias = r.get('via_ways') or [] 
        rtype = r.get('type', '')
        if rtype.startswith('no_'):
            banned_rels.append((r['from'], list(vias), r['to'], rtype))
        elif rtype.startswith('only_'):
            only_rels.append((r['from'], list(vias), r['to'], rtype))

    OUT_RELS = ('to', 'turns_into', 'merges_into', 'u_turns_into')
    LEN_KEY = {
        'to': 'to_lengths',
        'turns_into': 'turns_into_lengths',
        'merges_into': 'merges_into_lengths',
        'u_turns_into': 'u_turns_into_lengths',
    }

    def classify_direction(ego_id, tgt_id):
        delta = (segment_data[tgt_id]['entry_heading']
                 - segment_data[ego_id]['exit_heading'] + 180) % 360 - 180
        if -config["angle_straight_thresh"] <= delta <= config["angle_straight_thresh"]:
            return 'straight'
        if config["angle_straight_thresh"] < delta < config["angle_left_right_limit_propagate"]:
            return 'left'
        if -config["angle_left_right_limit_propagate"] < delta < -config["angle_straight_thresh"]:
            return 'right'
        if config["angle_u_turn_thresh"] <= abs(delta) <= 180:
            return 'u_turn'
        return 'straight'

    def follow_junction_chain(j_seed, visited=None, depth=0, max_depth=None):
        if max_depth is None:
            max_depth = config["junction_max_depth"]
        """
        Starting from a junction link j_seed, walk through every outgoing
        relation (to / turns_into / merges_into / u_turns_into) until a
        non-junction segment is reached.

        Returns a list of (exit_segment_id, full_chain, chain_rels) tuples
        where chain_rels is the sequence of relation keys traversed
        (one shorter than full_chain; empty if the first hop already
        exits the junction).
        """
        if visited is None:
            visited = set()
        if j_seed in visited or depth > max_depth or j_seed not in adjacency:
            return []
        visited = visited | {j_seed}
        results = []
        for relkey in OUT_RELS:
            neighbors = adjacency[j_seed].get(relkey, [])
            if relkey == 'to' and neighbors:
                neighbors = [neighbors[0]]
            for nxt in neighbors:
                if nxt in junction_set:
                    for tgt, chain, rels in follow_junction_chain(
                            nxt, visited, depth + 1, max_depth):
                        results.append((tgt, [j_seed] + chain, [relkey] + rels))
                else:
                    if nxt in adjacency:
                        results.append((nxt, [j_seed], [relkey]))
        return results


    # Gather new connections before mutating adjacency.
    additions = {}  # ego_id -> list of (tgt, new_rel, chain)
    for ego_id, adj in adjacency.items():
        if ego_id in junction_set:
            continue
        ego_osmids = parse_osmids(segment_data[ego_id]['osmid'])
        ego_osmids_set = set(ego_osmids)
        additions.setdefault(ego_id, [])

        for entry_rel in OUT_RELS:
            neighbors = adj.get(entry_rel, [])
            if entry_rel == 'to' and neighbors:
                neighbors = [neighbors[0]]
            for j in list(neighbors):
                if j not in junction_set:
                    continue
                for tgt, chain, chain_rels in follow_junction_chain(j):
                    if tgt == ego_id:
                        continue

                    chain_way_sets = [
                        set(parse_osmids(segment_data[c]['osmid'])) for c in chain
                    ]
                    tgt_osmids_set = set(parse_osmids(segment_data[tgt]['osmid']))
                    
                    path_way_sets = [ego_osmids_set] + chain_way_sets + [tgt_osmids_set]
                    
                    banned = False
                    forced_turn = False
                    
                    # Check only_* restrictions
                    for (f_id, vias, t_id, rtype) in only_rels:
                        seq = [f_id] + vias
                        curr_idx = -1
                        found_seq = True
                        for wid in seq:
                            found = False
                            for i in range(curr_idx + 1, len(path_way_sets)):
                                if wid in path_way_sets[i]:
                                    curr_idx = i
                                    found = True
                                    break
                            if not found:
                                found_seq = False
                                break
                                
                        if found_seq:
                            last_seq_wid = seq[-1]
                            violation = False
                            obeyed = False
                            for i in range(curr_idx + 1, len(path_way_sets)):
                                if last_seq_wid in path_way_sets[i]:
                                    continue
                                if t_id in path_way_sets[i]:
                                    obeyed = True
                                    break # Obeyed
                                else:
                                    violation = True
                                    break
                            
                            if violation:
                                turn_restrictions_logger.info(
                                    f"Junction propagation: [{ego_id}] -> [{tgt}] via {chain} "
                                    f"dropped by ONLY restriction ({rtype}) from {f_id} requiring {t_id}.")
                                banned = True
                                break
                            elif obeyed:
                                if 'straight' not in rtype:
                                    forced_turn = True
                                
                    if banned:
                        continue

                    # Check no_* restrictions
                    for (f_id, vias, t_id, rtype) in banned_rels:
                        seq = [f_id] + vias + [t_id]
                        curr_idx = -1
                        found_seq = True
                        for wid in seq:
                            found = False
                            for i in range(curr_idx + 1, len(path_way_sets)):
                                if wid in path_way_sets[i]:
                                    curr_idx = i
                                    found = True
                                    break
                            if not found:
                                found_seq = False
                                break
                        if found_seq:
                            turn_restrictions_logger.info(
                                f"Junction propagation: [{ego_id}] -> [{tgt}] via {chain} "
                                f"dropped by NO restriction ({rtype}).")
                            banned = True
                            break
                                
                    if banned:
                        continue

                    direction = classify_direction(ego_id, tgt)
                    all_through_after_entry = all(r == 'to' for r in chain_rels)
                    all_through = (entry_rel == 'to' and all_through_after_entry)
                    saw_merge = (entry_rel == 'merges_into'
                                 or 'merges_into' in chain_rels)

                    if direction == 'u_turn':
                        is_same_way = any(e_id in tgt_osmids_set for e_id in ego_osmids)
                        hw_type = segment_data[ego_id]['highway']
                        banned_hw = config["banned_u_turn_hw_types"]
                        is_banned_hw = hw_type in banned_hw
                        
                        v_node = segment_data[ego_id]['v']
                        out_degree = len(edges_by_u.get(v_node, []))
                        in_degree = len(edges_by_v.get(v_node, []))
                        is_mid_block = (out_degree <= 2 and in_degree <= 2)
                        
                        if is_same_way or is_banned_hw or is_mid_block:
                            junctions_logger.info(
                                f"Junction propagation: [{ego_id}] -> [{tgt}] via {chain} "
                                f"dropped by U-Turn Filters.")
                            continue
                            
                        new_rel = 'u_turns_into'
                    elif forced_turn:
                        new_rel = 'turns_into'
                    elif direction == 'straight' and all_through:
                        new_rel = 'to'
                    elif (entry_rel == 'merges_into' and all_through_after_entry) or (direction == 'straight' and saw_merge):
                        new_rel = 'merges_into'
                    else:
                        new_rel = 'turns_into'

                    if new_rel == 'turns_into':
                        ego_name = segment_data[ego_id]['name']
                        tgt_name = segment_data[tgt]['name']
                        if ego_name and tgt_name and ego_name == tgt_name:
                            junctions_logger.info(
                                f"Junction propagation: [{ego_id}] -> [{tgt}] via {chain} "
                                f"dropped (segment cannot turn_into the same road name '{ego_name}').")
                            continue

                    additions[ego_id].append((tgt, new_rel, chain))
                    junctions_logger.info(
                        f"Junction bypass: [{ego_id}] -{new_rel}-> [{tgt}] via {chain}.")

    # Apply: drop junction-link references from non-junction segments, then add new ones.
    for ego_id in adjacency.keys():
        if ego_id in junction_set:
            continue
        for relkey in OUT_RELS:
            old_list = adjacency[ego_id].get(relkey, [])
            old_lens = adjacency[ego_id].get(LEN_KEY[relkey], [])
            kept = [(t, l) for t, l in zip(old_list, old_lens) if t not in junction_set]
            adjacency[ego_id][relkey] = [t for t, _ in kept]
            adjacency[ego_id][LEN_KEY[relkey]] = [l for _, l in kept]
        old_from = adjacency[ego_id].get('from', [])
        old_from_lens = adjacency[ego_id].get('from_lengths', [])
        kept_from = [(f, l) for f, l in zip(old_from, old_from_lens) if f not in junction_set]
        adjacency[ego_id]['from'] = [f for f, _ in kept_from]
        adjacency[ego_id]['from_lengths'] = [l for _, l in kept_from]

    # Pick one relation per (ego, tgt) pair. Multiple junction chains from the
    # same ego can land on the same target with different classifications
    # (e.g. a through 'to' chain plus a branching chain that ends in
    # merges_into). The through-movement is the more specific fact, so we
    # apply this priority: to > merges_into > turns_into > u_turns_into.
    REL_PRIORITY = {'to': 0, 'merges_into': 1, 'turns_into': 2, 'u_turns_into': 3}
    for ego_id, conns in list(additions.items()):
        best_per_tgt = {}  # tgt -> (rank, new_rel, chain)
        for tgt, new_rel, chain in conns:
            rank = REL_PRIORITY[new_rel]
            prev = best_per_tgt.get(tgt)
            if prev is None or rank < prev[0]:
                best_per_tgt[tgt] = (rank, new_rel, chain)
        additions[ego_id] = [(tgt, rel, chain) for tgt, (_, rel, chain) in best_per_tgt.items()]

    for ego_id, conns in additions.items():
        seen = set()
        for tgt, new_rel, _chain in conns:
            key = (tgt, new_rel)
            if key in seen:
                continue
            seen.add(key)

            if new_rel == 'to' and _chain:
                if 'to_junction_chain' not in adjacency[ego_id] or not adjacency[ego_id]['to_junction_chain']:
                    adjacency[ego_id]['to_junction_chain'] = _chain
                    adjacency[ego_id]['to_junction_chain_lengths'] = [segment_data[c]['length'] for c in _chain]
                    
                if tgt in adjacency:
                    if 'from_junction_chain' not in adjacency[tgt] or not adjacency[tgt]['from_junction_chain']:
                        adjacency[tgt]['from_junction_chain'] = _chain[::-1]
                        adjacency[tgt]['from_junction_chain_lengths'] = [segment_data[c]['length'] for c in _chain[::-1]]

            # Reciprocal 'from' chain for straight through-connections.
            if new_rel == 'to' and tgt in adjacency:
                if ego_id not in adjacency[tgt].get('from', []):
                    adjacency[tgt].setdefault('from', []).append(ego_id)
                    adjacency[tgt].setdefault('from_lengths', []).append(
                        segment_data[ego_id]['length'])

            if tgt in adjacency[ego_id][new_rel]:
                continue
            adjacency[ego_id][new_rel].append(tgt)
            adjacency[ego_id][LEN_KEY[new_rel]].append(segment_data[tgt]['length'])

    # ---- Remove redundant bypass connections when a slip _link exists ----
    for ego_id in [sid for sid in adjacency.keys() if sid not in junction_set]:
        ego_outs = adjacency[ego_id].get('to', []) + adjacency[ego_id].get('turns_into', []) + adjacency[ego_id].get('merges_into', [])
        link_outs = [sid for sid in ego_outs if sid in segment_data and segment_data[sid]['highway'].endswith('_link')]
        
        targets_to_remove = set()
        for L in link_outs:
            L_targets = adjacency.get(L, {}).get('to', []) + adjacency.get(L, {}).get('turns_into', []) + adjacency.get(L, {}).get('merges_into', []) + adjacency.get(L, {}).get('u_turns_into', [])
            for S in L_targets:
                targets_to_remove.add(S)
                
        if targets_to_remove:
            for S in targets_to_remove:
                for relkey in ['to', 'turns_into', 'merges_into', 'u_turns_into']:
                    if S in adjacency[ego_id].get(relkey, []):
                        idx = adjacency[ego_id][relkey].index(S)
                        adjacency[ego_id][relkey].pop(idx)
                        len_key = relkey + '_lengths' if not relkey.endswith('s') else relkey[:-1] + '_lengths'
                        if len_key in adjacency[ego_id]:
                            adjacency[ego_id][len_key].pop(idx)
                        junctions_logger.info(f"Post-propagation cleanup: Removed redundant connection {ego_id} -> {S} because {ego_id} -> _link -> {S} exists.")
                        
                        if relkey == 'to' and S in adjacency and ego_id in adjacency[S].get('from', []):
                            f_idx = adjacency[S]['from'].index(ego_id)
                            adjacency[S]['from'].pop(f_idx)
                            if 'from_lengths' in adjacency[S]:
                                adjacency[S]['from_lengths'].pop(f_idx)

    # ---- Recompute crosses / crossed_by using the expanded junction footprint ----
    junctions_logger.info("Recomputing crosses/crossed_by after junction propagation...")

    nonjunc_ids = [sid for sid in adjacency.keys() if sid not in junction_set]

    enters_node = {}
    for sid in nonjunc_ids:
        v = segment_data[sid]['v']
        enters_node.setdefault(v, set()).add(sid)

    def junction_footprint(ego_id):
        """Nodes belonging to the junction(s) downstream of ego (inclusive of ego_v)."""
        nodes = {segment_data[ego_id]['v']}
        seen_junc = set()
        stack = []
        for relkey in OUT_RELS:
            neighbors = adjacency[ego_id].get(relkey, [])
            if relkey == 'to' and neighbors:
                neighbors = [neighbors[0]]
            for nxt in neighbors:
                if nxt in junction_set:
                    stack.append(nxt)
        while stack:
            j = stack.pop()
            if j in seen_junc:
                continue
            seen_junc.add(j)
            nodes.add(segment_data[j]['u'])
            nodes.add(segment_data[j]['v'])
            for relkey in OUT_RELS:
                neighbors = adjacency[j].get(relkey, [])
                if relkey == 'to' and neighbors:
                    neighbors = [neighbors[0]]
                for nxt in neighbors:
                    if nxt in junction_set and nxt not in seen_junc:
                        stack.append(nxt)
        return nodes

    for ego_id in nonjunc_ids:
        adj = adjacency[ego_id]
        turn_targets = (set(adj.get('turns_into', []))
                        | set(adj.get('merges_into', []))
                        | set(adj.get('u_turns_into', [])))
        through_target = set(adj.get('to', [])[:1])
        opposite = set(adj.get('opposite_direction', []))

        candidates = set()
        for n in junction_footprint(ego_id):
            candidates |= enters_node.get(n, set())
        candidates.discard(ego_id)
        candidates -= opposite

        crossed_by = []
        crosses = []
        for b in candidates:
            b_adj = adjacency[b]
            b_outs = set()
            for rk in OUT_RELS:
                b_outs |= set(b_adj.get(rk, []))
            if b_outs & turn_targets:
                if b not in crosses:
                    crosses.append(b)
            if (ego_id in b_outs) or (b_outs & turn_targets) or (b_outs & through_target):
                if b not in crossed_by:
                    crossed_by.append(b)

        adjacency[ego_id]['crossed_by'] = crossed_by
        adjacency[ego_id]['crossed_by_lengths'] = [
            segment_data[b]['length'] for b in crossed_by
        ]
        adjacency[ego_id]['crosses'] = crosses
        adjacency[ego_id]['crosses_lengths'] = [
            segment_data[b]['length'] for b in crosses
        ]
        
    # ---- Additional Topological Check for crosses / crossed_by ----
    for ego_id in nonjunc_ids:
        targets = adjacency[ego_id].get('turns_into', []) + adjacency[ego_id].get('u_turns_into', []) + adjacency[ego_id].get('merges_into', [])
        for tgt in targets:
            from_list = adjacency.get(tgt, {}).get('from', [])
            if from_list:
                from_id = from_list[0]
                if from_id != ego_id and from_id in nonjunc_ids:
                    if from_id not in adjacency[ego_id]['crosses']:
                        adjacency[ego_id]['crosses'].append(from_id)
                        adjacency[ego_id]['crosses_lengths'].append(segment_data[from_id]['length'])
                    if ego_id not in adjacency[from_id]['crossed_by']:
                        adjacency[from_id]['crossed_by'].append(ego_id)
                        adjacency[from_id]['crossed_by_lengths'].append(segment_data[ego_id]['length'])

    # ---- Calculate merges_with and merged_by ----
    for ego_id in nonjunc_ids:
        adjacency[ego_id]['merges_with'] = []
        adjacency[ego_id]['merges_with_lengths'] = []
        adjacency[ego_id]['merged_by'] = []
        adjacency[ego_id]['merged_by_lengths'] = []

    for ego_id in nonjunc_ids:
        for A in adjacency[ego_id].get('merges_into', []):
            if A in adjacency and A in nonjunc_ids:
                from_segs = adjacency[A].get('from', [])
                if from_segs:
                    P = from_segs[0]
                    if P in nonjunc_ids:
                        if P not in adjacency[ego_id]['merges_with']:
                            adjacency[ego_id]['merges_with'].append(P)
                            adjacency[ego_id]['merges_with_lengths'].append(segment_data[P]['length'])
                        if ego_id not in adjacency[P]['merged_by']:
                            adjacency[P]['merged_by'].append(ego_id)
                            adjacency[P]['merged_by_lengths'].append(segment_data[ego_id]['length'])

    # Same mutual-exclusion rule as in the initial build: drop any crosses/
    # crossed_by entries that are already captured by merges_with/merged_by.
    _dedupe_crosses_against_merges(adjacency)
    _extract_mutual_intersections(adjacency)

    junctions_logger.info(
        f"Junction propagation complete: rewired {sum(len(v) for v in additions.values())} "
        f"connections across {len(junction_set)} junction links.")

    return adjacency
    
def simplify_network_topology(gdf, adj):
    """
    Merge consecutive non-junction segments with simple 1-to-1 topological connections.
    This behaves like OSMnx simplify_graph but operates on final topology,
    prioritizing two-way segments' properties when merging.
    """
    logger.info("Simplifying final network topology (merging 1-to-1 non-junction chains)...")
    gdf = gdf.copy()
    gdf.set_index('segment_id', inplace=True, drop=False)

    # Ensure the merge-tracking column exists. Segments that absorb a neighbour
    # are flagged so build_controllers won't reassign them as junction links.
    if 'was_merged' not in gdf.columns:
        gdf['was_merged'] = 'False'
    
    edges_by_u = gdf.groupby('u').apply(lambda x: [str(i) for i in x.index]).to_dict()
    edges_by_v = gdf.groupby('v').apply(lambda x: [str(i) for i in x.index]).to_dict()
    
    total_merged = 0
    
    while True:
        found_merge = None
        
        for a_id in sorted(list(gdf.index.astype(str))):
            if a_id not in adj: continue
            if str(gdf.at[a_id, 'is_internal_junction']).lower() == 'true': continue
            
            a_to = adj[a_id].get('to', [])
            if len(a_to) != 1: continue
            b_id = str(a_to[0])
            
            if b_id not in gdf.index: continue
            if str(gdf.at[b_id, 'is_internal_junction']).lower() == 'true': continue
            
            a_v = gdf.at[a_id, 'v']
            b_u = gdf.at[b_id, 'u']
            if a_v != b_u: continue

            # Only merge segments with matching lane counts
            a_lanes = gdf.at[a_id, 'lanes'] if 'lanes' in gdf.columns else np.nan
            b_lanes = gdf.at[b_id, 'lanes'] if 'lanes' in gdf.columns else np.nan
            if pd.notna(a_lanes) and pd.notna(b_lanes) and float(a_lanes) != float(b_lanes):
                continue

            a_opp = [str(x) for x in adj[a_id].get('opposite_direction', [])]
            b_opp = [str(x) for x in adj[b_id].get('opposite_direction', [])]
            
            allowed_in = {a_id} | set(b_opp)
            allowed_out = {b_id} | set(a_opp)
            
            actual_in = set(edges_by_v.get(a_v, []))
            actual_out = set(edges_by_u.get(a_v, []))
            
            # Ensure no other topological edges connect at the shared node
            if not actual_in.issubset(allowed_in): continue
            if not actual_out.issubset(allowed_out): continue
            
            out_targets = set(adj[a_id].get('to', []) + adj[a_id].get('turns_into', []) + adj[a_id].get('merges_into', []) + adj[a_id].get('u_turns_into', []))
            if len(out_targets) != 1: continue
            
            found_merge = (a_id, b_id)
            break
            
        if not found_merge:
            break
            
        a_id, b_id = found_merge
        a_row = gdf.loc[a_id]
        b_row = gdf.loc[b_id]
        
        a_oneway = str(a_row.get('oneway', 'False')).lower() in ['true', 'yes', '1']
        b_oneway = str(b_row.get('oneway', 'False')).lower() in ['true', 'yes', '1']
        
        if not b_oneway and a_oneway:
            keeper = b_id
            absorber = a_id
            is_pred_keeper = False
        else:
            keeper = a_id
            absorber = b_id
            is_pred_keeper = True
            
        a_coords = list(a_row['geometry'].coords)
        b_coords = list(b_row['geometry'].coords)
        new_coords = a_coords + b_coords[1:] if a_coords[-1] == b_coords[0] else a_coords + b_coords
        merged_geom = LineString(new_coords)
        merged_len = merged_geom.length
        
        gdf.at[keeper, 'geometry'] = merged_geom
        gdf.at[keeper, 'length'] = merged_len
        
        a_osmids = parse_osmids(a_row.get('osmid', ''))
        b_osmids = parse_osmids(b_row.get('osmid', ''))
        combined_osmids = list(dict.fromkeys(a_osmids + b_osmids))
        if len(combined_osmids) == 1:
            gdf.at[keeper, 'osmid'] = combined_osmids[0]
        else:
            gdf.at[keeper, 'osmid'] = str(combined_osmids)
            
        if is_pred_keeper:
            gdf.at[keeper, 'v'] = b_row['v']
            if absorber in edges_by_u.get(b_row['u'], []): edges_by_u[b_row['u']].remove(absorber)
            if absorber in edges_by_v.get(b_row['v'], []): edges_by_v[b_row['v']].remove(absorber)
            if keeper in edges_by_v.get(a_row['v'], []): edges_by_v[a_row['v']].remove(keeper)
            edges_by_v.setdefault(b_row['v'], []).append(keeper)
            
            for rel in ['to', 'turns_into', 'merges_into', 'u_turns_into', 'banned_successors', 'only_successors', 'opposite_direction', 'merges_with']:
                adj[keeper][rel] = adj[absorber].get(rel, [])
                len_key = rel + '_lengths' if not rel.endswith('s') else rel[:-1] + '_lengths'
                if len_key in adj[absorber]:
                    adj[keeper][len_key] = adj[absorber][len_key]
                    
            for rel in ['crosses', 'crossed_by', 'intersects_with']:
                if rel == 'crossed_by': len_key = 'crossed_by_lengths'
                elif rel == 'crosses': len_key = 'crosses_lengths'
                else: len_key = 'intersects_with_lengths'
                existing = set(adj[keeper].get(rel, []))
                for i, item in enumerate(adj[absorber].get(rel, [])):
                    if item not in existing:
                        existing.add(item)
                        adj[keeper].setdefault(rel, []).append(item)
                        if len_key in adj[absorber] and len(adj[absorber][len_key]) > i:
                            adj[keeper].setdefault(len_key, []).append(adj[absorber][len_key][i])
        else:
            gdf.at[keeper, 'u'] = a_row['u']
            if absorber in edges_by_u.get(a_row['u'], []): edges_by_u[a_row['u']].remove(absorber)
            if absorber in edges_by_v.get(a_row['v'], []): edges_by_v[a_row['v']].remove(absorber)
            if keeper in edges_by_u.get(b_row['u'], []): edges_by_u[b_row['u']].remove(keeper)
            edges_by_u.setdefault(a_row['u'], []).append(keeper)
            
            adj[keeper]['from'] = adj[absorber].get('from', [])
            if 'from_lengths' in adj[absorber]:
                adj[keeper]['from_lengths'] = adj[absorber]['from_lengths']
                
            adj[keeper]['merged_by'] = adj[absorber].get('merged_by', [])
            if 'merged_by_lengths' in adj[absorber]:
                adj[keeper]['merged_by_lengths'] = adj[absorber]['merged_by_lengths']
                
            for rel in ['crosses', 'crossed_by', 'intersects_with']:
                if rel == 'crossed_by': len_key = 'crossed_by_lengths'
                elif rel == 'crosses': len_key = 'crosses_lengths'
                else: len_key = 'intersects_with_lengths'
                existing = set(adj[keeper].get(rel, []))
                for i, item in enumerate(adj[absorber].get(rel, [])):
                    if item not in existing:
                        existing.add(item)
                        adj[keeper].setdefault(rel, []).append(item)
                        if len_key in adj[absorber] and len(adj[absorber][len_key]) > i:
                            adj[keeper].setdefault(len_key, []).append(adj[absorber][len_key][i])
                            
        del adj[absorber]
        gdf = gdf.drop(index=absorber)
        gdf.at[keeper, 'was_merged'] = 'True'

        # Update adjacency references seamlessly avoiding a costly rebuild
        for sid, data in adj.items():
            for key, val_list in data.items():
                if isinstance(val_list, list) and absorber in val_list:
                    if key.endswith('_lengths') or 'junction_chain' in key:
                        continue
                        
                    len_key = key + '_lengths' if not key.endswith('s') else key[:-1] + '_lengths'
                    if key == 'crosses': len_key = 'crosses_lengths'
                    if key == 'crossed_by': len_key = 'crossed_by_lengths'
                    if key == 'intersects_with': len_key = 'intersects_with_lengths'
                    
                    has_lens = len_key in data and len(data[len_key]) == len(val_list)
                    
                    new_list = []
                    new_lens = []
                    for i, item in enumerate(val_list):
                        tgt_id = keeper if item == absorber else item
                        if tgt_id not in new_list:
                            new_list.append(tgt_id)
                            if has_lens:
                                if tgt_id == keeper:
                                    new_lens.append(merged_len)
                                else:
                                    new_lens.append(data[len_key][i])
                    data[key] = new_list
                    if has_lens:
                        data[len_key] = new_lens
                        
        total_merged += 1
        logger.info(f"Simplified non-junction pair: kept {keeper}, absorbed {absorber}.")
        
    logger.info(f"Simplified {total_merged} pairs of non-junction segments.")
    gdf.reset_index(drop=True, inplace=True)
    return gdf, adj


def main():
    parser = argparse.ArgumentParser(description="Build robust topological adjacency. Always downloads the latest network.")
    parser.add_argument('--output', default='topological_adjacency.json', help="Output JSON filename.")
    parser.add_argument('--bbox', type=float, nargs=4, default=[23.71317, 37.97161, 23.74515, 37.99880], help="Bounding box coordinates in the order: west south east north (left bottom right top)")
    parser.add_argument('--merge-short', action='store_true', help="Enable merging of segments shorter than classify_max_length_m to fill gaps")
    parser.add_argument('--merge-junctions', action='store_true', help="After all other steps, merge chained junction link pairs connected only by to/from, then re-run junction propagation")
    parser.add_argument('--simplify-network', action='store_true', help="Merge consecutive non-junction segments with simple 1-to-1 topological connections")
    args = parser.parse_args()
    
    config = {
        "valid_highway_types": [
            'primary', 'secondary', 'tertiary', 'trunk', 'residential', 
            'unclassified', 'living_street', 'road', 'primary_link', 
            'secondary_link', 'tertiary_link', 'trunk_link', 'motorway', 
            'motorway_link'
        ],
        "lane_prop_iterations": 10,
        "heading_look_dist": 10.0,
        "overpass_buffer": 0.001,
        "overpass_timeout": 40,
        "turn_restrictions_path": "turn_restrictions.json",
        "target_dist": 55.0,
        "angle_straight_thresh": 25,
        "angle_u_turn_thresh": 170,
        "angle_pred_limit": 75,
        "angle_same_way_thresh": 60.0,
        "angle_diff_way_thresh": 45.0,
        "angle_parallel_thresh": 60.0,
        "angle_left_right_limit_basic": 120,
        "angle_left_right_limit_propagate": 170,
        "banned_u_turn_hw_types": [
            'primary', 'secondary', 'trunk', 'motorway', 
            'primary_link', 'secondary_link', 'trunk_link', 'motorway_link'
        ],
        "classify_max_area_sqm": 350.0,
        "classify_max_length_m": 18.0,
        "classify_shape_max_length_m": 35.0,
        "classify_min_intersection_length": 1e-5,
        "segments_junctions_path": "segments_junctions.json",
        "junction_max_depth": 10,
        "network_path": "osm_network.gpkg",
        "signal_endpoint_tolerance": 1.0,
        "signal_split_offset": 3.5,
        "single_signal_split_threshold": 15.0,
        "controller_junction_tolerance": 5.0
    }
    
    # Always download the network as requested
    edges, signal_data = get_osm_network(tuple(args.bbox), config)
    signal_points = [s['geometry'] for s in signal_data]
    
    # 1. Build initial adjacency to know who connects to who
    adjacency, restrictions = build_topological_adjacency(edges, config)
    edges = classify_junctions(edges, config)

    # 2. Propagate topology through classified junction links so external segments
    #    connect directly to the segments on the far side of the junction.
    adjacency = propagate_junction_topology(adjacency, edges, restrictions, config)


    if args.merge_junctions:
            # 3. Greedy pairwise merge of chained junction links, then fully rebuild
            #    classification + adjacency + junction propagation so chains reflect
            #    the simplified geometry.
            edges = merge_junction_pairs(edges, adjacency, config)
            adjacency, restrictions = build_topological_adjacency(edges, config)
            edges = classify_junctions(edges, config)
            adjacency = propagate_junction_topology(adjacency, edges, restrictions, config)
            
    if args.merge_short:
        # 4. Merge standalone junction links
        edges = merge_short_segments(edges, adjacency, config)
        
    if args.simplify_network:
        # 5. Simplify non-junction 1-to-1 chains
        edges, adjacency = simplify_network_topology(edges, adjacency)

    # 6. Tag approach segments that terminate at a traffic-signal node. Done after all
    #    merges so the endpoint geometry reflects the final, persisted edge layout.
    edges, signal_data = tag_signals_at_end(edges, signal_data, adjacency, tolerance=config["signal_endpoint_tolerance"])

    # 6b. Split segments with multiple traffic signals
    edges, adjacency = split_multi_signal_segments(edges, adjacency, config)
    edges = classify_junctions(edges, config)

    # 6c. Unflag long _link segments with a single signal so they are not bypassed
    edges = unflag_long_link_junctions(edges, config)

    # 6d. Rebuild adjacency so predecessor/successor relationships reflect the final junction flags
    adjacency, restrictions = build_topological_adjacency(edges, config)

    adjacency = propagate_junction_topology(adjacency, edges, restrictions, config)

    # 7. Build logical signal controllers from signalized junctions.
    controllers, edges = build_controllers(edges, signal_data, adjacency, config)
    adjacency = propagate_junction_topology(adjacency, edges, restrictions, config)
    


    controllers_path = os.path.join(os.path.dirname(args.output) or '.', 'controllers.json')
    with open(controllers_path, 'w') as f:
        json.dump(controllers, f, indent=4)
    logger.info(f"Saved {len(controllers)} controllers to {controllers_path}")

    network_path = config["network_path"]
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