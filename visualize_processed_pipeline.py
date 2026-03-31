import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import json
import numpy as np

def visualize_processed_pipeline(network_file, processed_file, show_all=False, time_window=5.0):
    print(f"Loading data...")
    if not os.path.exists(network_file):
        print(f"Error: Network file {network_file} not found.")
        return
    if not os.path.exists(processed_file):
        print(f"Error: Processed file {processed_file} not found. Run the pipeline first.")
        return

    # Load Road Network for context
    edges = gpd.read_file(network_file)
    if edges.crs != "EPSG:4326":
        edges = edges.to_crs("EPSG:4326")

    df = pd.read_csv(processed_file)
    
    # Identify ALL processed vehicles before any sampling to filter background data correctly
    all_processed_vehicles = df['track_id'].unique()

    # ----------------------------------------------------
    # Load additional matched trajectories for grey points
    # ----------------------------------------------------
    matched_file = os.path.join(os.path.dirname(processed_file), 'matched_trajectories.csv')
    if not os.path.exists(matched_file):
        print(f"Warning: matched_trajectories.csv not found at {matched_file}. Grey points will not be shown.")
        df_matched = pd.DataFrame()
    else:
        print(f"Loading matched trajectories from {matched_file}...")
        # Only need specific columns to reduce memory load
        df_matched = pd.read_csv(matched_file, usecols=['track_id', 'time', 'lat', 'lon'])
        
        # Filter out vehicles that are in processed.csv
        df_matched = df_matched[~df_matched['track_id'].isin(all_processed_vehicles)]
        
        # Group by time window
        df_matched['time_window_sec'] = (df_matched['time'] // time_window) * time_window
        
        # Sampling logic for background trajectories
        if not show_all:
            unique_matched = df_matched['track_id'].unique()
            # Sample 10% for performance when not using --all
            n_sample_matched = max(1, int(len(unique_matched) * 0.10))
            print(f"Sampling 10% of background trajectories ({n_sample_matched} vehicles)...")
            sampled_matched = np.random.choice(unique_matched, n_sample_matched, replace=False)
            df_matched = df_matched[df_matched['track_id'].isin(sampled_matched)]
        else:
            print("Visualizing 100% of background trajectories.")

        # Downsample to 1 point per vehicle per frame for performance
        print("Downsampling matched trajectories (grey points)...")
        df_matched = df_matched.groupby(['time_window_sec', 'track_id']).first().reset_index()
        df_matched['time_window_str'] = df_matched['time_window_sec'].astype(int).astype(str) + "s"

    # ----------------------------------------------------
    # Process CAV Data
    # ----------------------------------------------------
    if not show_all:
        n_sample = max(1, int(len(all_processed_vehicles) * 0.25)) 
        print(f"Sampling 25% of CAVs ({n_sample} vehicles)...")
        sampled_tracks = np.random.choice(all_processed_vehicles, n_sample, replace=False)
        df = df[df['track_id'].isin(sampled_tracks)]
    else:
        print("Visualizing 100% of CAV trajectories.")

    df['track_id_str'] = df['track_id'].astype(str)
    # Ensure speed is float for heatmap
    if 'speed' in df.columns:
        df['speed'] = df['speed'].astype(float)

    # Group timestamps into buckets for the animation frame
    df['time_window_sec'] = (df['time'] // time_window) * time_window

    # PERFORMANCE FIX: Downsample to 1 point per vehicle per time window.      
    print(f"Optimizing payload: Downsampling to 1 point per vehicle per frame...")
    df = df.groupby(['time_window_sec', 'track_id']).first().reset_index()     

    df = df.sort_values(['time_window_sec', 'time'])
    df['time_window_str'] = df['time_window_sec'].astype(int).astype(str) + "s"

    # ----------------------------------------------------
    # Synchronize Timeline (Union of all buckets)
    # ----------------------------------------------------
    # Ensure the timeline slider covers all time steps from both datasets
    all_buckets = sorted(list(set(df['time_window_sec'].unique()) | (set(df_matched['time_window_sec'].unique()) if not df_matched.empty else set())))
    existing_cav_buckets = set(df['time_window_sec'].unique())
    missing_cav_buckets = [b for b in all_buckets if b not in existing_cav_buckets]
    
    if missing_cav_buckets:
        print(f"Aligning timeline: adding {len(missing_cav_buckets)} buckets found only in background data...")
        dummy_rows = pd.DataFrame({
            'time_window_sec': missing_cav_buckets,
            'time_window_str': [str(int(b)) + "s" for b in missing_cav_buckets],
            'lat': [None] * len(missing_cav_buckets),
            'lon': [None] * len(missing_cav_buckets),
            'track_id_str': ["dummy"] * len(missing_cav_buckets),
            'speed': [np.nan] * len(missing_cav_buckets)
        })
        df = pd.concat([df, dummy_rows], ignore_index=True)
    
    df = df.sort_values('time_window_sec')

    # Construct the hover template with all requested features
    hover_cols = {
        "track_id": True,
        "time": ":.2f",
        "relative_time_gap": ":.2f",
        "segment_id": True,
        "lane_index": True,
        "num_lanes": True,
        "speed": ":.2f",
        "relative_kinematic_ratio": ":.3f",
        # Occupancy
        "relative_occupancy_proceeding": ":.2f",
        "relative_occupancy_following": ":.2f",
        "relative_occupancy_leftwards_proceeding": ":.2f",
        "relative_occupancy_leftwards_following": ":.2f",
        "relative_occupancy_rightwards_proceeding": ":.2f",
        "relative_occupancy_rightwards_following": ":.2f",
        # Density
        "raw_density_proceeding": True,
        "raw_density_following": True,
        "raw_density_leftwards_proceeding": True,
        "raw_density_leftwards_following": True,
        "raw_density_rightwards_proceeding": True,
        "raw_density_rightwards_following": True,
        # Relative Speeds
        "relative_speed_proceeding": ":.2f",
        "relative_speed_following": ":.2f",
        "relative_speed_leftwards_proceeding": ":.2f",
        "relative_speed_leftwards_following": ":.2f",
        "relative_speed_rightwards_proceeding": ":.2f",
        "relative_speed_rightwards_following": ":.2f",
        # Lat/Lon hidden
        "lat": False,
        "lon": False,
        "time_window_sec": False,
        "time_window_str": False
    }

    print(f"Generating interactive map with {len(all_buckets)} time steps...")
    fig = px.scatter_map(
        df,
        lat="lat",
        lon="lon",
        color="speed",
        color_continuous_scale="Viridis",
        animation_frame="time_window_str",
        animation_group="track_id_str",
        hover_data=hover_cols,
        zoom=15,
        height=900,
        title=f"Pipeline Output Visualization ({'All' if show_all else 'Sampled'}, {time_window}s bins)"
    )

    fig.update_layout(showlegend=False)

    # Add background road network
    lats, lons = [], []
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
        line=dict(width=1.5, color='rgba(80, 80, 80, 0.25)'),
        hoverinfo='skip',
        showlegend=False
    )
    
    # Pre-calculate grey groups
    grey_frames = {}
    if not df_matched.empty:
        for t_str, group in df_matched.groupby('time_window_str'):
            grey_frames[str(t_str)] = group

    # Update frames to include grey points trace
    for frame in fig.frames:
        frame_time = frame.name
        if frame_time in grey_frames:
            g = grey_frames[frame_time]
            grey_trace = go.Scattermap(
                lat=g['lat'],
                lon=g['lon'],
                mode='markers',
                marker=dict(size=8, color='grey', opacity=0.5),
                hoverinfo='skip',
                showlegend=False
            )
        else:
            grey_trace = go.Scattermap(
                lat=[], lon=[], mode='markers', hoverinfo='skip', showlegend=False
            )
        
        frame.data = [grey_trace] + list(frame.data)
        frame.traces = [1, 2] # 0 is bg, 1 is grey, 2 is cavs

    # Prepare initial trace state
    first_frame_name = df['time_window_str'].iloc[0] if not df.empty else ""
    if first_frame_name in grey_frames:
        g = grey_frames[first_frame_name]
        initial_grey_trace = go.Scattermap(
            lat=g['lat'],
            lon=g['lon'],
            mode='markers',
            marker=dict(size=8, color='grey', opacity=0.5),
            hoverinfo='skip',
            showlegend=False
        )
    else:
        initial_grey_trace = go.Scattermap(
            lat=[], lon=[], mode='markers', hoverinfo='skip', showlegend=False
        )

    # Ensure background map is below the points
    fig.add_trace(bg_roads)
    fig.add_trace(initial_grey_trace)

    # Reorder initial traces: [Background Roads (0), Grey Points (1), CAV Heatmap (2)]
    data_list = list(fig.data)
    fig.data = tuple([data_list[-2], data_list[-1]] + data_list[:-2])
    cav_traces = data_list[:-2]
    
    # Explicitly set marker size for CAVs and other points to 8
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    
    # If px generated multiple traces for some reason, we update the frame.traces to match
    if len(cav_traces) > 1:
        for frame in fig.frames:
            frame.traces = [1] + list(range(2, 2 + len(cav_traces)))

    html_file = "pipeline_features_viz.html"
    print(f"Saving visualization to {html_file}...")
    fig.write_html(html_file)

    print(f"Opening {html_file} in your default web browser...")
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(html_file))

def main():
    parser = argparse.ArgumentParser(description="Visualize processed pipeline CAV features.")
    parser.add_argument('--network', default='osm_network.gpkg', help="Path to osm_network.gpkg")
    parser.add_argument('--time_window', type=float, default=5.0, help="Size of the timeline bucket in seconds (default: 5.0)")
    parser.add_argument('input', nargs='?', help="Path to the _processed.csv file")       
    parser.add_argument('--all', action='store_true', help="Visualize all points (no sampling)")
    args = parser.parse_args()

    if args.input:
        if not os.path.dirname(args.input):
            args.input = os.path.join('processed_data', args.input)
    else:
        search_dir = 'processed_data'
        if os.path.exists(search_dir):
            processed_files = [f for f in os.listdir(search_dir) if f.endswith('_processed.csv')]
            if processed_files:
                args.input = os.path.join(search_dir, processed_files[0])      
                print(f"No input specified, using: {args.input}")
            else:
                print(f"Error: No _processed.csv file found in '{search_dir}'. Please provide one with --input.")
                return
        else:
            print(f"Error: '{search_dir}' directory not found. Please provide a file with --input.")
            return

    visualize_processed_pipeline(args.network, args.input, args.all, args.time_window)

if __name__ == '__main__':
    main()
