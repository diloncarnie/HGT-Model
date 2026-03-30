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

    edges = gpd.read_file(network_file)
    if edges.crs != "EPSG:4326":
        edges = edges.to_crs("EPSG:4326")
        
    df = pd.read_csv(trajectory_file)
    
    if not test:
        print("Sampling 15% of vehicles...")
        unique_tracks = df['track_id'].unique()
        n_sample = max(1, int(len(unique_tracks) * 0.15))
        sampled_tracks = np.random.choice(unique_tracks, n_sample, replace=False)
        df = df[df['track_id'].isin(sampled_tracks)]
    else:
        print("Test mode: visualizing all trajectory points.")

    edges['segment_id_str'] = edges['segment_id'].astype(str)
    
    df['track_id_str'] = df['track_id'].astype(str)
    
    print("Generating base map...")
    
    # Create the scatter plot for trajectory points
    fig = px.scatter_map(
        df,
        lat="lat",
        lon="lon",
        color="track_id_str",
        custom_data=["segment_id"], # For JS highlighting
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
            "lat": False,
            "lon": False
        },
        zoom=15,
        height=800,
        title=f"Map Matching Visualization ({'All' if test else '15% Sampled'} Vehicles)"
    )
    fig.update_layout(showlegend=False)

    # Add ALL road lines as a faint background layer
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
        line=dict(width=2, color='rgba(100, 100, 100, 0.2)'),
        hoverinfo='skip',
        showlegend=False
    )
    fig.add_trace(bg_roads)
    
    # Reorder: roads at bottom
    data_list = list(fig.data)
    # Background roads should be first so points are on top
    fig.data = (data_list[-1], *data_list[:-1])

    print("Preparing interactive HTML...")
    
    # Map of segment_id -> coords for JS
    geo_map = {}
    for _, row in edges.iterrows():
        coords = []
        if row.geometry.geom_type == 'LineString':
            coords = [[y, x] for x, y in zip(*row.geometry.xy)]
        elif row.geometry.geom_type == 'MultiLineString':
            for line in row.geometry.geoms:
                coords.extend([[y, x] for x, y in zip(*line.xy)])
                coords.append([None, None])
        geo_map[row['segment_id_str']] = coords

    html_file = "map_matching_viz.html"
    base_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    
    js_injection = f"""
    <script>
        var geoMap = {json.dumps(geo_map)};
        
        document.addEventListener('DOMContentLoaded', function() {{
            var plotEl = document.getElementsByClassName('plotly-graph-div')[0];
            
            plotEl.on('plotly_click', function(data){{
                if(data.points.length > 0) {{
                    var segId = data.points[0].customdata[0];
                    console.log("Clicked Trajectory Point on Segment:", segId);
                    highlightSegment(segId, plotEl);
                }}
            }});
        }});

        function highlightSegment(segId, plotEl) {{
            var segIdStr = String(segId);
            
            var traces = [];
            
            if (geoMap[segIdStr]) {{
                traces.push({{
                    type: 'scattermap',
                    lat: geoMap[segIdStr].map(p => p[0]),
                    lon: geoMap[segIdStr].map(p => p[1]),
                    mode: 'lines',
                    line: {{width: 6, color: 'cyan'}},
                    name: 'Matched Segment: ' + segIdStr
                }});
            }}

            // Remove previous highlight traces (indices starting after the original data)
            // Original data has 1 road trace + N vehicle type traces
            var originalTraceCount = {len(fig.data)};
            var currentTracesCount = plotEl.data.length;
            var indicesToRemove = [];
            for (var i = originalTraceCount; i < currentTracesCount; i++) indicesToRemove.push(i);
            
            if (indicesToRemove.length > 0) {{
                Plotly.deleteTraces(plotEl, indicesToRemove);
            }}
            
            Plotly.addTraces(plotEl, traces);
        }}
    </script>
    </body>
    </html>
    """
    
    final_html = base_html.replace('</body>\n</html>', js_injection)
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"Opening {html_file} in your default web browser...")
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(html_file))

def main():
    parser = argparse.ArgumentParser(description="Visualize map-matched trajectory points.")
    parser.add_argument('--network', default='osm_network.gpkg', help="Path to osm_network.gpkg")
    parser.add_argument('--trajectories', default='processed_data/matched_trajectories.csv', help="Path to matched_trajectories.csv")
    parser.add_argument('--all', action='store_true', help="Visualize all points (no sampling)")
    args = parser.parse_args()
    
    visualize_map_matching(args.network, args.trajectories, args.all)

if __name__ == '__main__':
    main()
