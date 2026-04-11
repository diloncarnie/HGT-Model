import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import json
import numpy as np

def visualize_map_matching(network_file, trajectory_file, test=False):
    print(f"Loading data...")
    if not os.path.exists(network_file):
        print(f"Error: Network file {network_file} not found.")
        return
    if not os.path.exists(trajectory_file):
        print(f"Error: Trajectory file {trajectory_file} not found. Run extract_map_and_speeds.py first.")
        return

    # Load trajectories first to know the spatial extent and unique segments
    df = pd.read_csv(trajectory_file)
    
    if not test:
        print("Sampling 15% of vehicles...")
        unique_tracks = df['track_id'].unique()
        n_sample = max(1, int(len(unique_tracks) * 0.15))
        sampled_tracks = np.random.choice(unique_tracks, n_sample, replace=False)
        df = df[df['track_id'].isin(sampled_tracks)].copy()
    else:
        print("Visualizing all trajectory points.")
        df = df.copy()

    # Optimization: Assign 10 colors randomly and iteratively to vehicles
    unique_tracks = df['track_id'].unique()
    np.random.shuffle(unique_tracks)
    track_to_color = {tid: str(i % 10) for i, tid in enumerate(unique_tracks)}
    df['assigned_color'] = df['track_id'].map(track_to_color)
    df['track_id_str'] = df['track_id'].astype(str)
    
    # Generate 10 nice rainbow colors
    rainbow_palette = px.colors.sample_colorscale('turbo', np.linspace(0, 1, 10))

    # Load and optimize network data
    edges = gpd.read_file(network_file)
    if edges.crs != "EPSG:4326":
        edges = edges.to_crs("EPSG:4326")
    
    # Spatial pruning: Only keep roads within the bounding box of our trajectories
    # print("Pruning network to trajectory bounds...")
    # min_lon, min_lat, max_lon, max_lat = df['lon'].min(), df['lat'].min(), df['lon'].max(), df['lat'].max()
    # buffer = 0.005 # ~500m buffer
    # edges = edges.cx[min_lon-buffer:max_lon+buffer, min_lat-buffer:max_lat+buffer].copy()
    
    print("Generating base map...")
    
    # Create the scatter plot for trajectory points
    fig = px.scatter_map(
        df,
        lat="lat",
        lon="lon",
        color="assigned_color", # 10 traces instead of N
        color_discrete_sequence=rainbow_palette,
        hover_data={
            "track_id": True,
            "type": True,
            "speed": ":.2f",
            "time": ":.2f",
            "segment_id": True,
            "segment_length": ":.2f",
            "num_lanes": True,
            "lane_index": True,
            "t_proj": ":.2f",
            "prop_dist": ":.3f",
            "rel_heading": ":.2f",
            "assigned_color": False,
            "lat": False,
            "lon": False
        },
        zoom=15,
        height=800,
        title=f"Map Matching Visualization ({'All' if test else '15% Sampled'} Vehicles)"
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_maps(
        style="white-bg",
        layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "Esri",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ]
            }
        ],
        pitch=0,        # Lock to 2D
        bearing=0,      # No rotation
    )

    # Add filtered road lines as a faint background layer
    lats = []
    lons = []
    for geom in edges.geometry:
        if geom.geom_type == 'LineString':
            x, y = geom.xy
            lats.extend(list(y) + [None])
            lons.extend(list(x) + [None])
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                x, y = line.xy
                lats.extend(list(y) + [None])
                lons.extend(list(x) + [None])

    bg_roads = go.Scattermap(
        lat=lats,
        lon=lons,
        mode='lines',
        line=dict(width=1.5, color='rgba(255, 255, 255, 0.4)'),
        hoverinfo='skip',
        showlegend=False
    )
    fig.add_trace(bg_roads)
    
    # Reorder: roads at bottom
    data_list = list(fig.data)
    fig.data = (data_list[-1], *data_list[:-1])

    html_file = "map_matching_viz.html"
    print(f"Saving to {html_file}...")
    fig.write_html(html_file, include_plotlyjs="cdn")

    print(f"Opening {html_file} in your default web browser...")
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(html_file))


def main():
    parser = argparse.ArgumentParser(description="Visualize map-matched trajectory points.")
    parser.add_argument('--network', default='osm_network_merged.gpkg', help="Path to osm_network.gpkg")
    parser.add_argument('--trajectories', help="Path to matched_trajectories.csv")
    parser.add_argument('--all', action='store_true', help="Visualize all points (no sampling)")
    args = parser.parse_args()
    
    visualize_map_matching(args.network, args.trajectories, args.all)

if __name__ == '__main__':
    main()
