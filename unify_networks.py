import argparse
import os
import json
import multiprocessing
import logging
import pandas as pd
import geopandas as gpd

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

def generate_missing_reports(gpkg_files, gdfs, unified_gdf, common_gdf, base_dir, config, logger):
    common_ids = set(common_gdf['segment_id'])
    all_ids = set(unified_gdf['segment_id'])
    not_common_ids = all_ids - common_ids
    
    missing_report = {str(seg_id): [] for seg_id in not_common_ids}
    for file_path, current_gdf in zip(gpkg_files, gdfs):
        if 'segment_id' in current_gdf.columns:
            file_ids = set(current_gdf['segment_id'])
            missing_from_file = not_common_ids - file_ids
            for seg_id in missing_from_file:
                missing_report[str(seg_id)].append(file_path)
                
    removed_segments = []
    for seg_id, files_missing in missing_report.items():
        if len(files_missing) > config["removal_threshold"]:
            removed_segments.append(seg_id)
            
    removed_segments_file = os.path.join(base_dir, config["removed_segments_file"])
    
    if os.path.exists(removed_segments_file):
        try:
            with open(removed_segments_file, 'r') as f:
                existing_removed = json.load(f)
                removed_segments.extend(existing_removed)
        except Exception as e:
            logger.warning(f"Warning: Could not read existing {removed_segments_file}: {e}")
            
    removed_segments = list(set(removed_segments))
    logger.info(f"Saving commonly missing segments to {removed_segments_file}...")
    with open(removed_segments_file, 'w') as f:
        json.dump(removed_segments, f, indent=4)
        
    filtered_missing_report = {k: v for k, v in missing_report.items() if k not in removed_segments}

    missing_report_file = os.path.join(base_dir, config["missing_report_file"])
    logger.info(f"Saving missing segments report to {missing_report_file}...")
    with open(missing_report_file, 'w') as f:
        json.dump(filtered_missing_report, f, indent=4)
        
    network_missing_counts = {}
    for seg_id, files_missing in filtered_missing_report.items():
        if len(files_missing) <= config["missing_count_threshold"]:
            for f_path in files_missing:
                if f_path not in network_missing_counts:
                    network_missing_counts[f_path] = {"count": 0, "segments": []}
                network_missing_counts[f_path]["count"] += 1
                network_missing_counts[f_path]["segments"].append(seg_id)
    
    network_missing_counts_file = os.path.join(base_dir, config["network_missing_counts_file"])
    logger.info(f"Saving networks missing common segments counts to {network_missing_counts_file}...")
    with open(network_missing_counts_file, 'w') as f:
        json.dump(network_missing_counts, f, indent=4)
        
    return removed_segments

def main():
    parser = argparse.ArgumentParser(description="Unify OSM network files from multiple cities into a single file")
    parser.add_argument('--folder', default="processed_data", help="Path to processing folder")
    parser.add_argument('--filter_files', action='store_true', help="Filter removed segments from _processed.csv and matched_trajectories.csv files")
    args = parser.parse_args()
    
    CONFIG = {
        "removal_threshold": 5,
        "missing_count_threshold": 3,
        "removed_segments_file": "removed_segments.json",
        "missing_report_file": "missing_segments.json",
        "network_missing_counts_file": "network_missing_counts.json",
        "output_unified_file": "osm_network.gpkg",
        "output_common_file": "osm_network_common.gpkg",
        "filtered_network_file": "osm_network_filtered.gpkg"
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
    
    if has_segment_id:
        removed_segments = generate_missing_reports(valid_gpkg_files, gdfs, unified_gdf, common_gdf, base_dir, CONFIG, logger)
        
        logger.info("Filtering individual networks based on removed_segments in parallel...")
        removed_set = set(removed_segments)
        
        filter_args = [(f, gdf, removed_set, args.filter_files, CONFIG) for f, gdf in zip(valid_gpkg_files, gdfs)]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            filter_results = pool.map(_filter_dataset, filter_args)
            
        filtered_gdfs = []
        for filtered_gdf, logs in filter_results:
            for log_msg in logs:
                logger.info(log_msg)
            filtered_gdfs.append(filtered_gdf)
            
        logger.info("Re-evaluating unified and common networks...")
        unified_gdf, common_gdf, _ = deduplicate_networks(filtered_gdfs, logger)

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
