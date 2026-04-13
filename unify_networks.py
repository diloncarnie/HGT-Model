import argparse
import os
import json
import ast
import pandas as pd
import geopandas as gpd

def main():
    
    parser = argparse.ArgumentParser(description="Unify OSM network files from multiple cities into a single file")
    parser.add_argument('--folder', default="processed_data", help="Path to processing folder")
    parser.add_argument('--filter_files', action='store_true', help="Filter removed segments from _processed.csv and matched_trajectories.csv files")
    args = parser.parse_args()
    
    base_dir = args.folder
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    # Find all osm_network.gpkg files in subdirectories of processed_data
    # Exclude the one in processed_data itself to avoid reading a previously unified file
    gpkg_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip the root directory itself
        if root == base_dir:
            continue
            
        for file in files:
            if file == "osm_network.gpkg":
                gpkg_files.append(os.path.join(root, file))

    if not gpkg_files:
        print(f"No osm_network.gpkg files found in subdirectories of {base_dir}.")
        return

    print(f"Found {len(gpkg_files)} network files to unify.")

    gdfs = []
    for file in gpkg_files:
        print(f"Reading {file}...")
        try:
            gdf = gpd.read_file(file)
            print(f"Found {len(gdf)} segments in {file}.")
            gdfs.append(gdf)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not gdfs:
        print("Failed to read any valid network files.")
        return

    print("\nConcatenating networks...")
    unified_gdf = pd.concat(gdfs, ignore_index=True)

    print(f"Total segments before deduplication: {len(unified_gdf)}")
    
    # Drop duplicates. 
    # 'segment_id' is the primary identifier for segments in this pipeline.
    if 'segment_id' in unified_gdf.columns:
        common_ids = set(gdfs[0]['segment_id'])
        for gdf in gdfs[1:]:
            if 'segment_id' in gdf.columns:
                common_ids.intersection_update(set(gdf['segment_id']))
        unified_gdf = unified_gdf.drop_duplicates(subset=['segment_id'])
        common_gdf = unified_gdf[unified_gdf['segment_id'].isin(common_ids)]
        
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
            if len(files_missing) > 5:
                removed_segments.append(seg_id)
                    
        removed_segments_file = os.path.join(base_dir, "removed_segments.txt")
        
        # Read existing to allow for manual additions and handle duplicates
        if os.path.exists(removed_segments_file):
            try:
                with open(removed_segments_file, 'r') as f:
                    content = f.read()
                    if '=' in content:
                        list_str = content.split('=', 1)[1].strip()
                        existing_removed = ast.literal_eval(list_str)
                        removed_segments.extend(existing_removed)
            except Exception as e:
                print(f"Warning: Could not read existing {removed_segments_file}: {e}")
                
        removed_segments = list(set(removed_segments))
        print(f"Saving commonly missing segments to {removed_segments_file}...")
        with open(removed_segments_file, 'w') as f:
            f.write(f'removed_segments = {json.dumps(removed_segments)}\n')
            
        # Filter out removed segments from missing_report
        filtered_missing_report = {k: v for k, v in missing_report.items() if k not in removed_segments}

        missing_report_file = os.path.join(base_dir, "missing_segments.json")
        print(f"Saving missing segments report to {missing_report_file}...")
        with open(missing_report_file, 'w') as f:
            json.dump(filtered_missing_report, f, indent=4)
            
        network_missing_counts = {}
        for seg_id, files_missing in filtered_missing_report.items():
            if len(files_missing) <= 3:
                for f_path in files_missing:
                    if f_path not in network_missing_counts:
                        network_missing_counts[f_path] = {"count": 0, "segments": []}
                    network_missing_counts[f_path]["count"] += 1
                    network_missing_counts[f_path]["segments"].append(seg_id)
            
        network_missing_counts_file = os.path.join(base_dir, "network_missing_counts.json")
        print(f"Saving networks missing common segments counts to {network_missing_counts_file}...")
        with open(network_missing_counts_file, 'w') as f:
            json.dump(network_missing_counts, f, indent=4)
            
        # Filter the original networks and save them
        print("\nFiltering individual networks based on removed_segments...")
        removed_set = set(removed_segments)
        filtered_gdfs = []
        for file_path, current_gdf in zip(gpkg_files, gdfs):
            if 'segment_id' in current_gdf.columns:
                filtered_gdf = current_gdf[~current_gdf['segment_id'].astype(str).isin(removed_set)]
                filtered_gdfs.append(filtered_gdf)
                
                filtered_file_path = os.path.join(os.path.dirname(file_path), "osm_network_filtered.gpkg")
                print(f"Saving filtered network to {filtered_file_path}...")
                filtered_gdf.to_file(filtered_file_path, driver="GPKG")
            else:
                filtered_gdfs.append(current_gdf)
                
            if args.filter_files:
                dir_path = os.path.dirname(file_path)
                folder_name = os.path.basename(dir_path)
                
                traj_file = os.path.join(dir_path, "matched_trajectories.csv")
                if os.path.exists(traj_file):
                    print(f"Filtering {traj_file}...")
                    traj_df = pd.read_csv(traj_file)
                    if 'segment_id' in traj_df.columns:
                        filtered_traj = traj_df[~traj_df['segment_id'].astype(str).isin(removed_set)]
                        traj_out = os.path.join(dir_path, "matched_trajectories_filtered.csv")
                        filtered_traj.to_csv(traj_out, index=False)
                        
                proc_file = os.path.join(dir_path, f"{folder_name}_processed.csv")
                if os.path.exists(proc_file):
                    print(f"Filtering {proc_file}...")
                    proc_df = pd.read_csv(proc_file)
                    if 'segment_id' in proc_df.columns:
                        filtered_proc = proc_df[~proc_df['segment_id'].astype(str).isin(removed_set)]
                        proc_out = os.path.join(dir_path, f"{folder_name}_processed_filtered.csv")
                        filtered_proc.to_csv(proc_out, index=False)
                
        # Update unified_gdf and common_gdf
        print("\nRe-evaluating unified and common networks...")
        unified_gdf = pd.concat(filtered_gdfs, ignore_index=True)
        unified_gdf = unified_gdf.drop_duplicates(subset=['segment_id'])
        
        common_ids = set(filtered_gdfs[0]['segment_id']) if 'segment_id' in filtered_gdfs[0].columns else set()
        for gdf in filtered_gdfs[1:]:
            if 'segment_id' in gdf.columns:
                common_ids.intersection_update(set(gdf['segment_id']))
                
        common_gdf = unified_gdf[unified_gdf['segment_id'].isin(common_ids)]
    elif 'u' in unified_gdf.columns and 'v' in unified_gdf.columns:
        # Fallback to node pairs if segment_id isn't available
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
    else:
        # Final fallback to geometry
        unified_gdf = unified_gdf.drop_duplicates(subset=['geometry'])
        common_gdf = unified_gdf

    print(f"Total unique segments after deduplication: {len(unified_gdf)}")
    print(f"Total common segments present in all networks: {len(common_gdf)}")

    output_file = os.path.join(base_dir, "osm_network.gpkg")
    print(f"Saving unified network to {output_file}...")
    
    # Export unified network
    unified_gdf.to_file(output_file, driver="GPKG")
    
    common_output_file = os.path.join(base_dir, "osm_network_common.gpkg")
    print(f"Saving common network to {common_output_file}...")
    common_gdf.to_file(common_output_file, driver="GPKG")
    
    print("Done! The unified and common networks are ready.")

if __name__ == "__main__":
    main()
