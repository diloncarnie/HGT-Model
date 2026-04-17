import argparse
import os
import json
import logging
from collections import defaultdict
import geopandas as gpd


def build_node_graph(gdf):
    """Build a mapping from rounded endpoint coordinates to the set of segment_ids
    that touch each node, and a mapping from segment_id to its (start, end) nodes."""
    node_to_segs = defaultdict(set)
    seg_to_nodes = {}

    for _, row in gdf.iterrows():
        seg_id = str(row['segment_id'])
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            coords = list(geom.coords)
        except Exception:
            continue
        if len(coords) < 2:
            continue
        start = (round(coords[0][0], 7), round(coords[0][1], 7))
        end = (round(coords[-1][0], 7), round(coords[-1][1], 7))
        node_to_segs[start].add(seg_id)
        node_to_segs[end].add(seg_id)
        seg_to_nodes[seg_id] = (start, end)

    return node_to_segs, seg_to_nodes


def find_dangling_single_edges(node_to_segs, seg_to_nodes, logger):
    """Identify segment_ids that are single-edge protrusions from the network.

    A segment is a single-edge protrusion if:
      - One of its endpoints is a degree-1 node (a dead-end / leaf).
      - The other endpoint has degree >= 3, meaning the segment connects
        directly back to a well-connected part of the network.

    Segments where the dead-end connects to a degree-2 node are part of a
    chain of 2+ edges extending outward and are NOT removed.
    """
    dangling_ids = set()

    for seg_id, (start, end) in seg_to_nodes.items():
        start_deg = len(node_to_segs[start])
        end_deg = len(node_to_segs[end])

        # Check if exactly one endpoint is a leaf (degree 1)
        if start_deg == 1 and end_deg >= 3:
            dangling_ids.add(seg_id)
        elif end_deg == 1 and start_deg >= 3:
            dangling_ids.add(seg_id)

    logger.info(f"Found {len(dangling_ids)} single-edge dangling segments to prune.")
    return dangling_ids


def main():
    parser = argparse.ArgumentParser(
        description="Prune single-edge dangling segments from the unified OSM network."
    )
    parser.add_argument(
        '--folder', default="processed_data",
        help="Path to the processing folder containing osm_network.gpkg"
    )
    parser.add_argument(
        '--input_file', default="osm_network.gpkg",
        help="Name of the input network file inside --folder"
    )
    args = parser.parse_args()

    base_dir = args.folder
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    # Set up logger
    log_file = os.path.join(base_dir, "prune-network.log")
    logger = logging.getLogger("prune_network")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    input_path = os.path.join(base_dir, args.input_file)
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Reading network from {input_path}...")
    gdf = gpd.read_file(input_path)
    logger.info(f"Loaded {len(gdf)} segments.")

    if 'segment_id' not in gdf.columns:
        logger.error("Network file has no 'segment_id' column. Cannot prune.")
        return

    # Build the node graph and find dangling single edges
    node_to_segs, seg_to_nodes = build_node_graph(gdf)
    logger.info(f"Built graph with {len(node_to_segs)} nodes and {len(seg_to_nodes)} edges.")

    # Degree distribution summary
    degree_counts = defaultdict(int)
    for node, segs in node_to_segs.items():
        degree_counts[len(segs)] += 1
    logger.info(f"Node degree distribution: {dict(sorted(degree_counts.items()))}")

    dangling_ids = find_dangling_single_edges(node_to_segs, seg_to_nodes, logger)

    if dangling_ids:
        for seg_id in sorted(dangling_ids):
            start, end = seg_to_nodes[seg_id]
            start_deg = len(node_to_segs[start])
            end_deg = len(node_to_segs[end])
            logger.info(f"  Pruning segment {seg_id} (endpoint degrees: {start_deg}, {end_deg})")

    # Write removed segments JSON
    removed_path = os.path.join(base_dir, "removed_segments_pruned.json")
    with open(removed_path, 'w') as f:
        json.dump(sorted(dangling_ids), f, indent=4)
    logger.info(f"Wrote {len(dangling_ids)} pruned segment IDs to {removed_path}")

    # Write pruned network (never overwrite the input)
    pruned_gdf = gdf[~gdf['segment_id'].astype(str).isin(dangling_ids)].copy()
    pruned_path = os.path.join(base_dir, "osm_network_pruned.gpkg")
    pruned_gdf.to_file(pruned_path, driver="GPKG")
    logger.info(f"Saved pruned network ({len(gdf)} -> {len(pruned_gdf)} segments) to {pruned_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
