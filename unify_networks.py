import argparse
import os
import json
import multiprocessing
import logging
from collections import Counter
import numpy as np
import pandas as pd
import geopandas as gpd
from map_matching import filter_hq_d_values, density_prioritized_sample, jenks_breaks_with_fallback

def _read_gpkg(file_path):
    logs = []
    logs.append(f"Reading {file_path}...")
    try:
        gdf = gpd.read_file(file_path)
        logs.append(f"Found {len(gdf)} segments in {file_path}.")
        return file_path, gdf, logs
    except Exception as e:
        logs.append(f"Error reading {file_path}: {e}")
        return file_path, None, logs

def _filter_dataset(args):
    file_path, current_gdf, removed_set, filter_files, config = args
    logs = []
    filtered_gdf = None
    
    if 'segment_id' in current_gdf.columns:
        filtered_gdf = current_gdf[~current_gdf['segment_id'].astype(str).isin(removed_set)]
        filtered_file_path = os.path.join(os.path.dirname(file_path), config["filtered_network_file"])
        logs.append(f"Saving filtered network to {filtered_file_path}...")
        filtered_gdf.to_file(filtered_file_path, driver="GPKG")
    else:
        filtered_gdf = current_gdf
        
    if filter_files:
        dir_path = os.path.dirname(file_path)
        folder_name = os.path.basename(dir_path)
        
        traj_file = os.path.join(dir_path, "matched_trajectories.csv")
        if os.path.exists(traj_file):
            logs.append(f"Filtering {traj_file}...")
            traj_df = pd.read_csv(traj_file)
            if 'segment_id' in traj_df.columns:
                filtered_traj = traj_df[~traj_df['segment_id'].astype(str).isin(removed_set)]
                traj_out = os.path.join(dir_path, "matched_trajectories_filtered.csv")
                filtered_traj.to_csv(traj_out, index=False)
                
        proc_file = os.path.join(dir_path, f"{folder_name}_processed.csv")
        if os.path.exists(proc_file):
            logs.append(f"Filtering {proc_file}...")
            proc_df = pd.read_csv(proc_file)
            if 'segment_id' in proc_df.columns:
                filtered_proc = proc_df[~proc_df['segment_id'].astype(str).isin(removed_set)]
                proc_out = os.path.join(dir_path, f"{folder_name}_processed_filtered.csv")
                filtered_proc.to_csv(proc_out, index=False)
                
    return filtered_gdf, logs

def deduplicate_networks(gdfs, logger):
    unified_gdf = pd.concat(gdfs, ignore_index=True)
    logger.info(f"Total segments before deduplication: {len(unified_gdf)}")
    
    if 'segment_id' in unified_gdf.columns:
        common_ids = set(gdfs[0]['segment_id'])
        for gdf in gdfs[1:]:
            if 'segment_id' in gdf.columns:
                common_ids.intersection_update(set(gdf['segment_id']))
        unified_gdf = unified_gdf.drop_duplicates(subset=['segment_id'])
        common_gdf = unified_gdf[unified_gdf['segment_id'].isin(common_ids)]
        return unified_gdf, common_gdf, True
    elif 'u' in unified_gdf.columns and 'v' in unified_gdf.columns:
        if 'key' in unified_gdf.columns:
            subset_cols = ['u', 'v', 'key']
        else:
            subset_cols = ['u', 'v']
            
        common_pairs = set(tuple(x) for x in gdfs[0][subset_cols].values)
        for gdf in gdfs[1:]:
            if all(c in gdf.columns for c in subset_cols):
                common_pairs.intersection_update(set(tuple(x) for x in gdf[subset_cols].values))
                
        unified_gdf = unified_gdf.drop_duplicates(subset=subset_cols)
        common_gdf = unified_gdf[unified_gdf.set_index(subset_cols).index.isin(common_pairs)]
        return unified_gdf, common_gdf, False
    else:
        unified_gdf = unified_gdf.drop_duplicates(subset=['geometry'])
        common_gdf = unified_gdf
        return unified_gdf, common_gdf, False

def _collect_presence_and_junctions(gdfs):
    """For each segment_id, count how many networks it appears in, and collect ids flagged as internal junctions in any network."""
    presence_count = Counter()
    internal_junction_ids = set()
    for gdf in gdfs:
        if 'segment_id' not in gdf.columns:
            continue
        ids = gdf['segment_id'].astype(str)
        presence_count.update(set(ids.tolist()))
        if 'is_internal_junction' in gdf.columns:
            mask = gdf['is_internal_junction'].astype(str).str.lower().eq('true').values
            internal_junction_ids.update(ids[mask].tolist())
    return presence_count, internal_junction_ids


def identify_rare_segments(gpkg_files, gdfs, base_dir, config, logger):
    """Flag segments that are too rare across networks to keep.

    A segment is considered rare if it appears in fewer than
    (1 - removal_threshold_fraction) * N networks. Segments flagged as
    internal junctions in any network are exempt from automatic removal
    (those are expected to differ between runs).

    Writes the internally-flagged set to removed_segments_unified.json.
    Never reads or writes removed_segments.json — that file is user-maintained.
    Returns (unified_removed, internal_junction_ids).
    """
    num_networks = len(gpkg_files)
    presence_count, internal_junction_ids = _collect_presence_and_junctions(gdfs)

    min_presence = (1 - config["removal_threshold_fraction"]) * num_networks
    rare_ids = {sid for sid, c in presence_count.items() if c < min_presence}
    newly_removed = rare_ids - internal_junction_ids

    presence_hist = Counter(presence_count.values())
    logger.info(f"Networks: {num_networks}, unique segments: {len(presence_count)}")
    logger.info(f"Presence histogram (networks_present -> num_segments): {dict(sorted(presence_hist.items()))}")
    logger.info(f"Internal junctions (exempt from removal): {len(internal_junction_ids)}")
    logger.info(f"Rarity threshold: keep if present in >= {min_presence:.2f} of {num_networks} networks")
    logger.info(f"Rare segments: {len(rare_ids)}; removed after junction exemption: {len(newly_removed)}")

    unified_path = os.path.join(base_dir, config["removed_segments_unified_file"])
    with open(unified_path, 'w') as f:
        json.dump(sorted(newly_removed), f, indent=4)
    logger.info(f"Wrote {len(newly_removed)} internally-flagged segments to {unified_path}")
    return newly_removed, internal_junction_ids


def load_manual_removed_segments(base_dir, config, logger):
    """Read the user-maintained removed_segments.json without modifying it.

    Returns a set of segment id strings, or an empty set if the file is absent.
    """
    removed_path = os.path.join(base_dir, config["removed_segments_file"])
    if os.path.exists(removed_path):
        try:
            with open(removed_path, 'r') as f:
                manual = {str(x) for x in json.load(f)}
            logger.info(f"Loaded {len(manual)} manually-defined removed segments from {removed_path}")
            return manual
        except Exception as e:
            logger.warning(f"Could not read {removed_path}: {e}")
    else:
        logger.info(f"No {removed_path} found — no manual removals applied.")
    return set()


def write_missing_reports(gpkg_files, gdfs, unified_gdf, common_gdf, removed_set, internal_junction_ids, base_dir, config, logger):
    """Write two diagnostic reports about segments that are not common to every network,
    excluding segments already flagged for removal. Internal-junction segments are
    included and tagged with "is_internal_junction": true so they can be inspected
    or filtered downstream.

    - missing_report_file: per-segment, which network files are missing it.
    - network_missing_counts_file: per-network, which near-common segments it is missing.
    """
    common_ids = set(common_gdf['segment_id'].astype(str))
    all_ids = set(unified_gdf['segment_id'].astype(str))
    not_common_ids = (all_ids - common_ids) - removed_set

    per_segment_missing = {sid: [] for sid in not_common_ids}
    for file_path, gdf in zip(gpkg_files, gdfs):
        if 'segment_id' not in gdf.columns:
            continue
        file_ids = set(gdf['segment_id'].astype(str))
        for sid in not_common_ids - file_ids:
            per_segment_missing[sid].append(file_path)

    per_segment = {
        sid: {
            "is_internal_junction": sid in internal_junction_ids,
            "missing_from": files_missing,
        }
        for sid, files_missing in per_segment_missing.items()
    }

    missing_path = os.path.join(base_dir, config["missing_report_file"])
    with open(missing_path, 'w') as f:
        json.dump(per_segment, f, indent=4)
    logger.info(f"Wrote per-segment missing report ({len(per_segment)} segments) to {missing_path}")

    per_network = {}
    for sid, files_missing in per_segment_missing.items():
        if len(files_missing) <= config["missing_count_threshold"]:
            is_junction = sid in internal_junction_ids
            for fp in files_missing:
                entry = per_network.setdefault(fp, {"count": 0, "segments": [], "junction_segments": []})
                entry["count"] += 1
                if is_junction:
                    entry["junction_segments"].append(sid)
                else:
                    entry["segments"].append(sid)

    counts_path = os.path.join(base_dir, config["network_missing_counts_file"])
    with open(counts_path, 'w') as f:
        json.dump(per_network, f, indent=4)
    logger.info(f"Wrote per-network missing counts to {counts_path}")

def determine_final_lanes(avg_lanes_data, threshold, logger):
    """Decide a unified lane count per segment from per-file detected lane counts.

    Rules (in order):
      1. If any single lane count appears in >= threshold of the detected_lanes_list, use it.
      2. Otherwise fall back to osm_defined_lanes if available.
      3. Otherwise use the (already-rounded) average_detected_lanes.
      4. As a last resort, default to 1.
    """
    final_lanes = {}
    decision_log = {}
    reason_counts = Counter()
    final_lane_dist = Counter()

    for seg_id, info in avg_lanes_data.items():
        detected_list = info.get("detected_lanes_list") or []
        avg_lanes = info.get("average_detected_lanes")
        osm_lanes = info.get("osm_defined_lanes")

        decision = {
            "detected_lanes_list": detected_list,
            "osm_defined_lanes": osm_lanes,
            "average_detected_lanes": avg_lanes,
        }

        chosen, reason = None, None

        if detected_list:
            counter = Counter(detected_list)
            most_common, count = counter.most_common(1)[0]
            ratio = count / len(detected_list)
            decision["consensus_lanes"] = int(most_common)
            decision["consensus_ratio"] = round(ratio, 3)

            if ratio >= threshold:
                chosen, reason = int(most_common), "consensus"

        if chosen is None:
            if osm_lanes is not None:
                chosen, reason = int(osm_lanes), "osm_fallback"
            elif avg_lanes is not None:
                chosen, reason = int(avg_lanes), "average_fallback"
            else:
                chosen, reason = 1, "default_one_lane"

        decision["final_lanes"] = chosen
        decision["reason"] = reason
        final_lanes[str(seg_id)] = chosen
        decision_log[str(seg_id)] = decision
        reason_counts[reason] += 1
        final_lane_dist[chosen] += 1

    logger.info(f"Lane decision breakdown: {dict(reason_counts)}")
    logger.info(f"Final lane count distribution: {dict(sorted(final_lane_dist.items()))}")
    return final_lanes, decision_log


def _apply_unified_lanes_worker(args):
    network_file_path, filtered_gdf, final_lanes, config = args
    logs = []

    dir_path = os.path.dirname(network_file_path)
    folder_name = os.path.basename(dir_path)
    traj_filtered = os.path.join(dir_path, "matched_trajectories_filtered.csv")
    traj_original = os.path.join(dir_path, "matched_trajectories.csv")
    traj_file = traj_filtered if os.path.exists(traj_filtered) else traj_original

    if not os.path.exists(traj_file):
        logs.append(f"[{folder_name}] No matched_trajectories.csv to update.")
        return logs

    try:
        df = pd.read_csv(traj_file)
    except Exception as e:
        logs.append(f"[{folder_name}] Error reading {traj_file}: {e}")
        return logs

    required_cols = {'segment_id', 'signed_dist', 'speed', 'rel_heading'}
    missing = required_cols - set(df.columns)
    if missing:
        logs.append(f"[{folder_name}] CSV missing columns {missing}; skipping.")
        return logs

    if 'lane_index' not in df.columns:
        df['lane_index'] = 0
    if 'is_outlier' not in df.columns:
        df['is_outlier'] = False
    if 'num_lanes' not in df.columns:
        df['num_lanes'] = 0
    if 'D' not in df.columns:
        df['D'] = 0.0

    df['segment_id'] = df['segment_id'].astype(str)
    new_lane_boundaries = {}
    segments_processed = 0

    for seg_id, group in df.groupby('segment_id'):
        if not seg_id or seg_id == "nan" or seg_id not in final_lanes or len(group) == 0:
            continue

        k = final_lanes[seg_id]
        signed = group['signed_dist'].values

        # Maintain 98th/2nd percentile clipping for robust width and outlier flagging
        upper = float(np.percentile(signed, 98))
        lower = float(np.percentile(signed, 2))
        road_width = max(upper - lower, 1e-6)
        d_vals = np.clip(upper - signed, 0.0, road_width)

        hq_d = filter_hq_d_values(d_vals, group['speed'].values, group['rel_heading'].values, config)
        if len(hq_d) == 0:
            hq_d = d_vals
        if len(hq_d) == 0:
            continue

        road_min, road_max = float(hq_d.min()), float(hq_d.max())
        sample = density_prioritized_sample(hq_d, config) if k > 1 else hq_d
        bounds = jenks_breaks_with_fallback(sample, k, road_min, road_max)
        new_lane_boundaries[seg_id] = bounds

        raw_indices = np.digitize(d_vals, bins=bounds)
        seg_outliers = (raw_indices == 0) | (raw_indices == len(bounds))

        df.loc[group.index, 'D'] = d_vals
        df.loc[group.index[seg_outliers], 'is_outlier'] = True
        df.loc[group.index, 'lane_index'] = np.clip(raw_indices, 1, len(bounds) - 1) - 1
        df.loc[group.index, 'num_lanes'] = k
        segments_processed += 1

    traj_out = os.path.join(dir_path, "matched_trajectories_filtered.csv")
    df.to_csv(traj_out, index=False)

    boundaries_file = os.path.join(dir_path, "lane_boundaries_unified.json")
    with open(boundaries_file, 'w') as f:
        json.dump(new_lane_boundaries, f, indent=4)

    logs.append(f"[{folder_name}] Re-applied unified lanes to {segments_processed} segments in matched_trajectories_filtered.csv.")

    if filtered_gdf is not None and 'segment_id' in filtered_gdf.columns:
        gpkg_path = os.path.join(dir_path, "osm_network_filtered.gpkg")
        try:
            gdf = filtered_gdf.copy()
            mapped = gdf['segment_id'].astype(str).map(final_lanes)
            if 'lanes' in gdf.columns:
                gdf['lanes'] = mapped.combine_first(gdf['lanes'])
            else:
                gdf['lanes'] = mapped
            gdf.to_file(gpkg_path, driver='GPKG')
            logs.append(f"[{folder_name}] Updated 'lanes' in osm_network_filtered.gpkg.")
        except Exception as e:
            logs.append(f"[{folder_name}] Error updating {gpkg_path}: {e}")

    return logs


def main():
    parser = argparse.ArgumentParser(description="Unify OSM network files from multiple cities into a single file")
    parser.add_argument('--folder', default="processed_data", help="Path to processing folder")
    parser.add_argument('--filter_files', action='store_true', help="Filter removed segments from _processed.csv and matched_trajectories.csv files")
    args = parser.parse_args()
    
    CONFIG = {
        "removal_threshold_fraction": 0.2,
        "missing_count_threshold": 3,
        "removed_segments_file": "removed_segments.json",
        "removed_segments_unified_file": "removed_segments_unified.json",
        "missing_report_file": "missing_segments.json",
        "network_missing_counts_file": "network_missing_counts.json",
        "output_unified_file": "osm_network.gpkg",
        "output_common_file": "osm_network_common.gpkg",
        "filtered_network_file": "osm_network_filtered.gpkg",
        "average_lanes_file": "average_detected_lanes.json",
        "final_lanes_file": "final_lanes.json",
        "lane_decisions_file": "lane_decisions.json",
        "consensus_threshold": 0.8,
        "jenks_speed_thresholds": [7.5, 5.0, 3.0, 1.0],
        "jenks_heading_threshold": 5.0,
        "jenks_min_points": 50,
        "jenks_target_per_bin": 10,
        "jenks_max_target_calc": 15000,
        "jenks_bins": 40,
    }
    
    base_dir = args.folder
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return
        
    # Set up global logger
    log_file = os.path.join(base_dir, "unify-networks.log")
    logger = logging.getLogger("unify_networks")
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

    # Find all osm_network.gpkg files in subdirectories of processed_data
    # Exclude the one in processed_data itself to avoid reading a previously unified file
    gpkg_files = []
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            continue
        for file in files:
            if file == "osm_network.gpkg":
                gpkg_files.append(os.path.join(root, file))

    if not gpkg_files:
        logger.warning(f"No osm_network.gpkg files found in subdirectories of {base_dir}.")
        return

    logger.info(f"Found {len(gpkg_files)} network files to unify.")

    num_processes = multiprocessing.cpu_count()
    gdfs = []
    valid_gpkg_files = []
    
    logger.info("Reading networks in parallel...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_read_gpkg, gpkg_files)
        
    for file_path, gdf, logs in results:
        for log_msg in logs:
            if "Error" in log_msg:
                logger.error(log_msg)
            else:
                logger.info(log_msg)
                
        if gdf is not None:
            valid_gpkg_files.append(file_path)
            gdfs.append(gdf)

    if not gdfs:
        logger.error("Failed to read any valid network files.")
        return

    logger.info("Concatenating networks...")
    unified_gdf, common_gdf, has_segment_id = deduplicate_networks(gdfs, logger)
    
    filtered_gdfs = list(gdfs)
    merged_removed = set()
    if has_segment_id:
        # Compute internally-flagged set → saved to removed_segments_unified.json only.
        # removed_segments.json is never read or written by this step.
        unified_removed, internal_junction_ids = identify_rare_segments(
            valid_gpkg_files, gdfs, base_dir, CONFIG, logger
        )

        # Load the user-maintained manual removals (file is never modified).
        manual_removed = load_manual_removed_segments(base_dir, CONFIG, logger)

        # Full merged set used for the unified/common network outputs.
        merged_removed = unified_removed | manual_removed
        logger.info(
            f"Merged removal set: {len(merged_removed)} segments "
            f"(unified-flagged: {len(unified_removed)}, manual: {len(manual_removed)})"
        )

        # Diagnostic reports reflect what will be excluded from the unified/common outputs.
        write_missing_reports(
            valid_gpkg_files, gdfs, unified_gdf, common_gdf,
            merged_removed, internal_junction_ids, base_dir, CONFIG, logger
        )

        # Individual filtered network GPKGs:
        #   - Without --filter_files: apply only the manual list (do not touch CSVs).
        #   - With    --filter_files: apply the merged list and update CSVs.
        network_filter_set = merged_removed if args.filter_files else manual_removed
        logger.info(
            f"Filtering individual networks using "
            f"{'merged' if args.filter_files else 'manual-only'} removal set "
            f"({len(network_filter_set)} segments)..."
        )
        filter_args = [
            (f, gdf, network_filter_set, args.filter_files, CONFIG)
            for f, gdf in zip(valid_gpkg_files, gdfs)
        ]
        with multiprocessing.Pool(processes=num_processes) as pool:
            filter_results = pool.map(_filter_dataset, filter_args)

        filtered_gdfs = []
        for filtered_gdf, logs in filter_results:
            for log_msg in logs:
                logger.info(log_msg)
            filtered_gdfs.append(filtered_gdf)

        # The unified and common outputs always exclude the full merged set.
        if merged_removed:
            logger.info(f"Applying {len(merged_removed)} merged removals to unified and common networks...")
            unified_gdf = unified_gdf[~unified_gdf['segment_id'].astype(str).isin(merged_removed)].copy()
            common_gdf = common_gdf[~common_gdf['segment_id'].astype(str).isin(merged_removed)].copy()

    avg_lanes_path = os.path.join(base_dir, CONFIG["average_lanes_file"])
    if os.path.exists(avg_lanes_path):
        logger.info(f"Reading {avg_lanes_path} for lane unification...")
        with open(avg_lanes_path, 'r') as f:
            avg_lanes_data = json.load(f)

        if has_segment_id and merged_removed:
            filtered_avg_lanes = {k: v for k, v in avg_lanes_data.items() if str(k) not in merged_removed}
            logger.info(f"Filtered average_detected_lanes: {len(avg_lanes_data)} -> {len(filtered_avg_lanes)} segments (excluded {len(avg_lanes_data) - len(filtered_avg_lanes)} removed)")
        else:
            filtered_avg_lanes = avg_lanes_data

        final_lanes, decision_log = determine_final_lanes(
            filtered_avg_lanes, CONFIG["consensus_threshold"], logger
        )

        final_lanes_path = os.path.join(base_dir, CONFIG["final_lanes_file"])
        logger.info(f"Saving final unified lanes to {final_lanes_path}...")
        with open(final_lanes_path, 'w') as f:
            json.dump({k: int(v) for k, v in final_lanes.items()}, f, indent=4)

        decisions_path = os.path.join(base_dir, CONFIG["lane_decisions_file"])
        logger.info(f"Saving lane decision log to {decisions_path}...")
        with open(decisions_path, 'w') as f:
            json.dump(decision_log, f, indent=4)

        if args.filter_files:
            logger.info("Re-applying unified lane assignments to each file in parallel...")
            apply_args = [
                (fp, filt_gdf, final_lanes, CONFIG)
                for fp, filt_gdf in zip(valid_gpkg_files, filtered_gdfs)
            ]
            with multiprocessing.Pool(processes=num_processes) as pool:
                apply_results = pool.map(_apply_unified_lanes_worker, apply_args)

            for log_list in apply_results:
                for msg in log_list:
                    logger.info(msg)
        else:
            logger.info("Skipping per-file lane re-application (use --filter_files to apply).")

        # Update unified and common gdfs in memory so the saved outputs reflect the final lanes
        unified_gdf = unified_gdf.copy()
        common_gdf = common_gdf.copy()
        for gdf in (unified_gdf, common_gdf):
            if 'segment_id' in gdf.columns:
                mapped = gdf['segment_id'].astype(str).map(final_lanes)
                if 'lanes' in gdf.columns:
                    gdf['lanes'] = mapped.combine_first(gdf['lanes'])
                else:
                    gdf['lanes'] = mapped
        logger.info("Updated unified and common networks with final lane counts.")
    else:
        logger.info(f"No {avg_lanes_path} found - skipping lane unification.")

    logger.info(f"Total unique segments after deduplication: {len(unified_gdf)}")
    logger.info(f"Total common segments present in all networks: {len(common_gdf)}")

    output_file = os.path.join(base_dir, CONFIG["output_unified_file"])
    logger.info(f"Saving unified network to {output_file}...")
    unified_gdf.to_file(output_file, driver="GPKG")
    
    common_output_file = os.path.join(base_dir, CONFIG["output_common_file"])
    logger.info(f"Saving common network to {common_output_file}...")
    common_gdf.to_file(common_output_file, driver="GPKG")
    
    logger.info("Done! The unified and common networks are ready.")

if __name__ == "__main__":
    main()
