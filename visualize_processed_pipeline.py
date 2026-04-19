import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import json
import numpy as np

def visualize_processed_pipeline(network_file, input_path, show_all=False, time_window=1.0):
    print(f"Loading data...")
    if not os.path.exists(network_file):
        print(f"Error: Network file {network_file} not found.")
        return
    if not os.path.exists(input_path):
        print(f"Error: Input path {input_path} not found. Run the pipeline first.")
        return

    # Load Road Network for context
    edges = gpd.read_file(network_file)
    if edges.crs != "EPSG:4326":
        edges = edges.to_crs("EPSG:4326")

    # Load controllers.json for traffic lights and controller centroids
    controllers_file = os.path.join(os.path.dirname(network_file), 'controllers.json')
    controllers = {}
    if os.path.exists(controllers_file):
        with open(controllers_file, 'r') as f:
            controllers = json.load(f)
        print(f"Loaded {len(controllers)} signal controllers from {controllers_file}.")
    else:
        print(f"Warning: controllers.json not found at {controllers_file}. Signals will not be shown.")

    signal_lats, signal_lons, signal_hover = [], [], []
    centroid_lats, centroid_lons, centroid_hover, centroid_ids = [], [], [], []

    for ctrl_id, ctrl in controllers.items():
        for sig in ctrl.get('raw_osm_signals', []):
            signal_lats.append(sig['lat'])
            signal_lons.append(sig['lon'])
            signal_hover.append(
                f"Traffic signal on segment {sig.get('hosted_on_segment', '?')} "
                f"(Controller: {ctrl_id})"
            )
            
        jc = ctrl.get('junction_centroid')
        if jc:
            centroid_lats.append(jc['lat'])
            centroid_lons.append(jc['lon'])
            centroid_ids.append(ctrl_id)
            n_approach = len(ctrl.get('approach_segments', []))
            n_signals = ctrl.get('signal_count', 0)
            centroid_hover.append(
                f"<b>Controller {ctrl_id}</b><br>"
                f"Source: {ctrl.get('source', '?')}<br>"
                f"Signals: {n_signals}<br>"
                f"Approaches: {n_approach}<br>"
                f"Segments: {', '.join(ctrl.get('approach_segments', []))}"
            )

    processed_dfs = []
    matched_dfs = []

    # Handle directory vs single file
    if os.path.isdir(input_path):
        print(f"Directory detected. Loading sequential data from {input_path}...")
        subfolders = sorted([f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))])
        
        cumulative_time_offset = 0.0
        
        for folder_name in subfolders:
            subdir = os.path.join(input_path, folder_name)
            
            # Find processed file
            proc_file = os.path.join(subdir, f"{folder_name}_processed.csv")
            if not os.path.exists(proc_file):
                proc_files = [f for f in os.listdir(subdir) if f.endswith('_processed.csv')]
                if proc_files:
                    proc_file = os.path.join(subdir, proc_files[0])
            
            if os.path.exists(proc_file):
                print(f"  Loading {proc_file}...")
                df_p = pd.read_csv(proc_file)
                all_processed = df_p['track_id'].unique()
                
                # Make track_id globally unique across sequential files
                df_p['track_id'] = df_p['track_id'].astype(str) + f"_{folder_name}"
                
                # Load matched trajectories
                match_file = os.path.join(subdir, 'matched_trajectories_updated.csv')
                if not os.path.exists(match_file):
                    match_file = os.path.join(subdir, 'matched_trajectories.csv')
        
                if os.path.exists(match_file):
                    print(f"  Loading {match_file}...")
                    df_m = pd.read_csv(match_file, usecols=lambda c: c in ['track_id', 'time', 'lat', 'lon', 'is_outlier', 'is_parked'])
                    # Filter out processed vehicles for this specific folder
                    df_m = df_m[~df_m['track_id'].isin(all_processed)]
                    df_m['track_id'] = df_m['track_id'].astype(str) + f"_{folder_name}"
                else:
                    df_m = pd.DataFrame()
                    
                # Find max time in the current folder to offset the NEXT folder
                max_p_time = df_p['time'].max() if not df_p.empty else 0.0
                max_m_time = df_m['time'].max() if not df_m.empty else 0.0
                folder_max_time = max(max_p_time, max_m_time)
                
                # Shift timestamps
                if not df_p.empty:
                    df_p['time'] += cumulative_time_offset
                    processed_dfs.append(df_p)
                    
                if not df_m.empty:
                    df_m['time'] += cumulative_time_offset
                    matched_dfs.append(df_m)
                    
                if folder_max_time > 0:
                    cumulative_time_offset += folder_max_time
                    
        if not processed_dfs:
            print(f"Error: No processed CSV files found in subdirectories of {input_path}.")
            return
            
        df = pd.concat(processed_dfs, ignore_index=True)
        df_matched = pd.concat(matched_dfs, ignore_index=True) if matched_dfs else pd.DataFrame()
        
    else:
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        all_processed = df['track_id'].unique()
        df['track_id'] = df['track_id'].astype(str) # Ensure string format
        
      
        matched_file = os.path.join(os.path.dirname(input_path), 'matched_trajectories_updated.csv')
        if not os.path.exists(matched_file):
            matched_file = os.path.join(os.path.dirname(input_path), 'matched_trajectories.csv')
            
        if not os.path.exists(matched_file):
            print(f"Warning: matched_trajectories.csv not found at {matched_file}. Grey points will not be shown.")
            df_matched = pd.DataFrame()
        else:
            print(f"Loading matched trajectories from {matched_file}...")
            df_matched = pd.read_csv(matched_file, usecols=lambda c: c in ['track_id', 'time', 'lat', 'lon', 'is_outlier', 'is_parked'])
            # Filter out processed vehicles
            df_matched = df_matched[~df_matched['track_id'].isin(all_processed)]
            df_matched['track_id'] = df_matched['track_id'].astype(str)

    if 'is_outlier' not in df.columns:
        df['is_outlier'] = False
    if 'is_parked' not in df.columns:
        df['is_parked'] = False

    # ----------------------------------------------------
    # Downsample matched trajectories for grey points
    # ----------------------------------------------------
    df_matched_outliers = pd.DataFrame()
    df_matched_parked = pd.DataFrame()
    if not df_matched.empty:
        if 'is_outlier' not in df_matched.columns:
            df_matched['is_outlier'] = False
        if 'is_parked' not in df_matched.columns:
            df_matched['is_parked'] = False
            
        df_matched['time_window_sec'] = (df_matched['time'] // time_window) * time_window
        
        if not show_all:
            unique_matched = df_matched['track_id'].unique()
            n_sample_matched = max(1, int(len(unique_matched) * 0.10))
            print(f"Sampling 10% of background trajectories ({n_sample_matched} vehicles)...")
            sampled_matched = np.random.choice(unique_matched, n_sample_matched, replace=False)
            df_matched = df_matched[df_matched['track_id'].isin(sampled_matched)]
        else:
            print("Visualizing 100% of background trajectories.")

        print("Downsampling matched trajectories (grey points)...")
        df_matched_parked = df_matched[df_matched['is_parked']].copy()
        df_matched = df_matched[~df_matched['is_parked']].copy()
        
        df_matched_outliers = df_matched[df_matched['is_outlier']].copy()
        df_matched = df_matched[~df_matched['is_outlier']].copy()
        
        df_matched = df_matched.groupby(['time_window_sec', 'track_id']).first().reset_index()
        df_matched['time_window_str'] = df_matched['time_window_sec'].astype(int).astype(str) + "s"
        
        if not df_matched_outliers.empty:
            df_matched_outliers = df_matched_outliers.groupby(['time_window_sec', 'track_id']).first().reset_index()
            df_matched_outliers['time_window_str'] = df_matched_outliers['time_window_sec'].astype(int).astype(str) + "s"

        if not df_matched_parked.empty:
            df_matched_parked = df_matched_parked.groupby(['time_window_sec', 'track_id']).first().reset_index()
            df_matched_parked['time_window_str'] = df_matched_parked['time_window_sec'].astype(int).astype(str) + "s"

    # ----------------------------------------------------
    # Process CAV Data
    # ----------------------------------------------------
    all_processed_vehicles = df['track_id'].unique()
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

    df_parked = df[df['is_parked']].copy()
    df = df[~df['is_parked']].copy()

    df_outliers = df[df['is_outlier']].copy()
    df = df[~df['is_outlier']].copy()

    # PERFORMANCE FIX: Downsample to 1 point per vehicle per time window.      
    print(f"Optimizing payload: Downsampling to 1 point per vehicle per frame...")
    df = df.groupby(['time_window_sec', 'track_id']).first().reset_index()     

    df = df.sort_values(['time_window_sec', 'time'])
    df['time_window_str'] = df['time_window_sec'].astype(int).astype(str) + "s"

    if not df_outliers.empty:
        df_outliers = df_outliers.groupby(['time_window_sec', 'track_id']).first().reset_index()
        df_outliers = df_outliers.sort_values(['time_window_sec', 'time'])
        df_outliers['time_window_str'] = df_outliers['time_window_sec'].astype(int).astype(str) + "s"

    if not df_parked.empty:
        df_parked = df_parked.groupby(['time_window_sec', 'track_id']).first().reset_index()
        df_parked = df_parked.sort_values(['time_window_sec', 'time'])
        df_parked['time_window_str'] = df_parked['time_window_sec'].astype(int).astype(str) + "s"
        
    all_outliers = pd.concat([df_outliers, df_matched_outliers], ignore_index=True) if not df_outliers.empty or not df_matched_outliers.empty else pd.DataFrame()
    outlier_frames = {}
    if not all_outliers.empty:
        for t_str, group in all_outliers.groupby('time_window_str'):
            outlier_frames[str(t_str)] = group
    
    all_parked = pd.concat([df_parked, df_matched_parked], ignore_index=True) if not df_parked.empty or not df_matched_parked.empty else pd.DataFrame()
    parked_frames = {}
    if not all_parked.empty:
        for t_str, group in all_parked.groupby('time_window_str'):
            parked_frames[str(t_str)] = group

    # ----------------------------------------------------
    # Synchronize Timeline (Union of all buckets)
    # ----------------------------------------------------
    # Ensure the timeline slider covers all time steps from both datasets
    all_buckets_set = set(df['time_window_sec'].unique())
    if not df_matched.empty:
        all_buckets_set |= set(df_matched['time_window_sec'].unique())
    if not all_outliers.empty:
        all_buckets_set |= set(all_outliers['time_window_sec'].unique())
    if not all_parked.empty:
        all_buckets_set |= set(all_parked['time_window_sec'].unique())
        
    all_buckets = sorted(list(all_buckets_set))
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
        "is_outlier": True,
        "is_parked": True,
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
    
    signal_markers = go.Scattermap(
        lat=signal_lats,
        lon=signal_lons,
        mode='markers',
        marker=dict(size=8, color='red', symbol='circle'),
        text=signal_hover,
        hoverinfo='text',
        name=f'Traffic Signals ({len(signal_lats)})',
        showlegend=True
    )

    controller_centroids = go.Scattermap(
        lat=centroid_lats,
        lon=centroid_lons,
        mode='markers',
        marker=dict(size=18, color='gold', symbol='circle', opacity=0.7),
        text=centroid_hover,
        customdata=centroid_ids,
        hoverinfo='text',
        name=f'Controllers ({len(centroid_lats)})',
        showlegend=True,
    )

    # Pre-calculate grey groups
    grey_frames = {}
    if not df_matched.empty:
        for t_str, group in df_matched.groupby('time_window_str'):
            grey_frames[str(t_str)] = group

    # Update frames to include grey and outlier points traces
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
            
        if frame_time in outlier_frames:
            g_out = outlier_frames[frame_time]
            outlier_trace = go.Scattermap(
                lat=g_out['lat'], lon=g_out['lon'], mode='markers',
                marker=dict(size=8, color='darkred', opacity=0.9),
                hovertext=g_out['track_id'].astype(str),
                hoverinfo='text', showlegend=False
            )
        else:
            outlier_trace = go.Scattermap(
                lat=[], lon=[], mode='markers', hoverinfo='skip', showlegend=False
            )
        
        if frame_time in parked_frames:
            g_parked = parked_frames[frame_time]
            parked_trace = go.Scattermap(
                lat=g_parked['lat'], lon=g_parked['lon'], mode='markers',
                marker=dict(size=8, color='blue', opacity=0.9),
                hovertext=g_parked['track_id'].astype(str) + " (Parked)",
                hoverinfo='text', showlegend=False
            )
        else:
            parked_trace = go.Scattermap(
                lat=[], lon=[], mode='markers', hoverinfo='skip', showlegend=False
            )
        
        frame.data = [grey_trace, outlier_trace, parked_trace] + list(frame.data)

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
        
    if first_frame_name in outlier_frames:
        g_out = outlier_frames[first_frame_name]
        initial_outlier_trace = go.Scattermap(
            lat=g_out['lat'], lon=g_out['lon'], mode='markers',
            marker=dict(size=8, color='darkred', opacity=0.9),
            hovertext=g_out['track_id'].astype(str),
            hoverinfo='text', showlegend=False
        )
    else:
        initial_outlier_trace = go.Scattermap(
            lat=[], lon=[], mode='markers', hoverinfo='skip', showlegend=False
        )

    if first_frame_name in parked_frames:
        g_parked = parked_frames[first_frame_name]
        initial_parked_trace = go.Scattermap(
            lat=g_parked['lat'], lon=g_parked['lon'], mode='markers',
            marker=dict(size=8, color='blue', opacity=0.9),
            hovertext=g_parked['track_id'].astype(str) + " (Parked)",
            hoverinfo='text', showlegend=False
        )
    else:
        initial_parked_trace = go.Scattermap(
            lat=[], lon=[], mode='markers', hoverinfo='skip', showlegend=False
        )

    # Ensure background map is below the points
    fig.add_trace(bg_roads)
    fig.add_trace(signal_markers)
    fig.add_trace(controller_centroids)
    fig.add_trace(initial_grey_trace)
    fig.add_trace(initial_outlier_trace)
    fig.add_trace(initial_parked_trace)

    # Reorder initial traces: [Background Roads (0), Signals (1), Controllers (2), Grey Points (3), White Outliers (4), Blue Parked (5), CAV Heatmap (6+)]
    data_list = list(fig.data)
    fig.data = tuple(data_list[-6:] + data_list[:-6])
    cav_traces = data_list[:-6]
    
    # Explicitly set marker size for CAVs and other points to 8
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    
    # Restore controller marker size
    fig.update_traces(marker=dict(size=18), selector=dict(name=f'Controllers ({len(centroid_lats)})'))
    
    # If px generated multiple traces for some reason, we update the frame.traces to match
    for frame in fig.frames:
        frame.traces = [3, 4, 5] + list(range(6, 6 + len(cav_traces)))

    html_file = "pipeline_features_viz.html"
    print(f"Saving visualization to {html_file}...")
    fig.write_html(html_file)

    print(f"Opening {html_file} in your default web browser...")
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(html_file))

def main():
    parser = argparse.ArgumentParser(description="Visualize processed pipeline CAV features.")
    parser.add_argument('--folder', required=True, help="Path to the folder containing osm_network.gpkg and processed CSV file")
    parser.add_argument('--time_window', type=float, default=1.0, help="Size of the timeline bucket in seconds (default: 5.0)")
    parser.add_argument('--all', action='store_true', help="Visualize all points (no sampling)")
    args = parser.parse_args()

    folder_name = os.path.basename(os.path.normpath(args.folder))
    network_path = os.path.join(args.folder, 'osm_network_filtered.gpkg')
    input_path = os.path.join(args.folder, f"{folder_name}_processed_filtered.csv")
    
    if not os.path.exists(network_path):
        network_path = os.path.join(args.folder, 'osm_network.gpkg')
    if not os.path.exists(input_path):
        input_path = os.path.join(args.folder, f"{folder_name}_processed.csv")
    
    visualize_processed_pipeline(network_path, input_path, args.all, args.time_window)

if __name__ == '__main__':
    main()
