import os
import pandas as pd
import geopandas as gpd

def main():
    base_dir = "processed_data"
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
        unified_gdf = unified_gdf.drop_duplicates(subset=['segment_id'])
    elif 'u' in unified_gdf.columns and 'v' in unified_gdf.columns:
        # Fallback to node pairs if segment_id isn't available
        if 'key' in unified_gdf.columns:
            unified_gdf = unified_gdf.drop_duplicates(subset=['u', 'v', 'key'])
        else:
            unified_gdf = unified_gdf.drop_duplicates(subset=['u', 'v'])
    else:
        # Final fallback to geometry
        unified_gdf = unified_gdf.drop_duplicates(subset=['geometry'])

    print(f"Total unique segments after deduplication: {len(unified_gdf)}")

    output_file = os.path.join(base_dir, "osm_network.gpkg")
    print(f"Saving unified network to {output_file}...")
    
    # Export unified network
    unified_gdf.to_file(output_file, driver="GPKG")
    print("Done! The unified network is ready.")

if __name__ == "__main__":
    main()
