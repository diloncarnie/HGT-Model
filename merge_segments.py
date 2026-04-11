import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import LineString
from build_adjacency import build_topological_adjacency
SEGMENT_THRESHOLD = 15
def main():
    print("Loading network data...")
    gdf = gpd.read_file("osm_network.gpkg")
    
    print("Loading topological adjacency...")
    with open("topological_adjacency.json", "r") as f:
        adj = json.load(f)

    # Ensure segment_id is a string and set as index for easy access
    gdf['segment_id'] = gdf['segment_id'].astype(str)
    gdf.set_index('segment_id', inplace=True, drop=False)
    
    # Union-Find redirect map for merged segments
    redirect = {}
    def get_target(t):
        while t in redirect:
            t = redirect[t]
        return t

    small_segments = gdf[gdf['length'] < SEGMENT_THRESHOLD].index.tolist()
    deleted = set()
    
    print(f"Found {len(small_segments)} segments shorter than the threshold")
    
    for s_id in small_segments:
        if s_id in deleted:
            continue
            
        # Check if length got increased beyond the threshold by a previous merge
        if gdf.at[s_id, 'length'] >= SEGMENT_THRESHOLD:
            continue
            
        s_adj = adj.get(str(s_id))
        if not s_adj:
            continue
            
        t_id_original = None
        is_succ = True
        
        # Prefer merging into the first successor
        if s_adj.get('successors'):
            t_id_original = s_adj['successors'][0]
            is_succ = True
        # Fallback to the first predecessor
        elif s_adj.get('predecessors'):
            t_id_original = s_adj['predecessors'][0]
            is_succ = False
            
        if not t_id_original:
            continue
            
        t_id = get_target(t_id_original)
        
        # Prevent self-loops in edge cases
        if t_id == s_id:
            continue
            
        if t_id not in gdf.index:
            continue
            
        # Perform the merge S into T
        s_row = gdf.loc[s_id]
        t_row = gdf.loc[t_id]
        
        if is_succ:
            # S -> T
            # S is (A, B). T is (B, ...). B is s_row['v']
            old_node = s_row['v']
            new_node = s_row['u']
            
            s_coords = list(s_row['geometry'].coords)
            t_coords = list(t_row['geometry'].coords)
            
            if s_coords[-1] == t_coords[0]:
                new_coords = s_coords + t_coords[1:]
            else:
                new_coords = s_coords + t_coords
        else:
            # T -> S
            # S is (B, C). T is (..., B). B is s_row['u']
            old_node = s_row['u']
            new_node = s_row['v']
            
            s_coords = list(s_row['geometry'].coords)
            t_coords = list(t_row['geometry'].coords)
            
            if t_coords[-1] == s_coords[0]:
                new_coords = t_coords + s_coords[1:]
            else:
                new_coords = t_coords + s_coords
                
        merged_geom = LineString(new_coords)
        
        # Update T's geometry and length
        gdf.at[t_id, 'geometry'] = merged_geom
        gdf.at[t_id, 'length'] = merged_geom.length
        
        # Edge contraction: replace all occurrences of old_node with new_node
        # This keeps the graph fully connected topologically
        gdf.loc[gdf['u'] == old_node, 'u'] = new_node
        gdf.loc[gdf['v'] == old_node, 'v'] = new_node
        
        deleted.add(s_id)
        redirect[s_id] = t_id
        
    print(f"Merged and marked {len(deleted)} small segments for deletion.")
    
    # Remove deleted segments from the GeoDataFrame
    gdf = gdf[~gdf.index.isin(deleted)]
    
    gdf.reset_index(drop=True, inplace=True)
    
    out_gpkg = "osm_network_merged.gpkg"
    print(f"Saving merged network to {out_gpkg}...")
    gdf.to_file(out_gpkg, driver="GPKG")
    
    print("Rebuilding topological adjacency map...")
    new_adj = build_topological_adjacency(gdf)
    
    out_json = "topological_adjacency_merged.json"
    print(f"Exporting updated adjacency to {out_json}...")
    with open(out_json, "w") as f:
        json.dump(new_adj, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()