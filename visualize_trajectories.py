import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import ast

def visualize(input_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    if df.empty:
        print("Error: The provided CSV file is empty.")
        return
        
    if 'segment_id' not in df.columns:
        raise ValueError(f"Error: 'segment_id' column not found in {input_file}. You likely provided a raw or '_long.csv' file instead of the final '_processed.csv' file.")

    # Load frozen OSM Network to map segment_id back to osmid
    input_dir = os.path.dirname(os.path.abspath(input_file))
    network_path = os.path.join(input_dir, "osm_network.gpkg")
    
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"Cannot find OSM network file at {network_path}. Please re-run preprocess_pneuma.py to generate it.")
        
    print(f"Loading frozen OSM network from {network_path}...")
    edges = gpd.read_file(network_path)
    
    # CRITICAL: Re-project to WGS84 for Plotly (original is UTM)
    if edges.crs != "EPSG:4326":
        print("Re-projecting road network to WGS84...")
        edges = edges.to_crs("EPSG:4326")

    osmid_map = {}
    for idx, row in edges.iterrows():
        way_id_str = str(row['osmid'])
        if way_id_str.startswith('['):
            try:
                way_list = ast.literal_eval(way_id_str)
                way_id = way_list[0] if way_list else way_id_str
            except (ValueError, SyntaxError):
                way_id = way_id_str.strip("[]").split(",")[0].strip()
        else:
            way_id = way_id_str
            
        osmid_map[str(row['segment_id'])] = way_id

    def get_osm_url(seg_id):
        way_id = osmid_map.get(str(seg_id))
        if way_id:
            return f'https://www.openstreetmap.org/way/{way_id}'
        return ""

    df['osm_url'] = df['segment_id'].apply(get_osm_url)
    df['track_id_str'] = df['track_id'].astype(str)

    # Prepare road segment lines for highlighting (Drawn once per segment)
    print("Preparing road segment highlight traces...")
    unique_segs = df['segment_id'].unique().astype(str)
    edges['segment_id_str'] = edges['segment_id'].astype(str)
    active_edges = edges[edges['segment_id_str'].isin(unique_segs)].copy()
    
    # Use a colorful qualitative palette for the segments
    seg_palette = px.colors.qualitative.Dark24
    
    road_traces = []
    # Group segments by color to keep the number of traces low
    for i, color in enumerate(seg_palette):
        # Assign every Nth segment to this color
        group = active_edges.iloc[i::len(seg_palette)]
        if group.empty:
            continue
            
        lats = []
        lons = []
        for geom in group.geometry:
            if geom.geom_type == 'LineString':
                x, y = geom.xy
                lats.extend(list(y) + [None])
                lons.extend(list(x) + [None])
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    x, y = line.xy
                    lats.extend(list(y) + [None])
                    lons.extend(list(x) + [None])
        
        trace = go.Scattermap(
            lat=lats,
            lon=lons,
            mode='lines',
            line=dict(width=16, color=color),
            opacity=0.25,
            hoverinfo='skip',
            showlegend=False,
            name=f"Road Group {i}"
        )
        road_traces.append(trace)

    # Define shared color map for trajectory points
    unique_tracks = df['track_id_str'].unique()
    track_palette = px.colors.qualitative.Alphabet
    track_color_map = {tid: track_palette[i % len(track_palette)] for i, tid in enumerate(unique_tracks)}

    print("Generating interactive Map visualization...")
    fig = px.scatter_map(
        df,
        lat="lat",
        lon="lon",
        color="track_id_str",
        color_discrete_map=track_color_map,
        custom_data=["osm_url"], 
        hover_data={
            "track_id_str": False,
            "track_id": True,
            "speed": ":.2f",
            "segment_id": True,
            "segment_length": True,
            "osm_url": False,
            "lane_index": True,
            "num_lanes": True,
            "raw_density_proceeding": True,
            "raw_density_following": True,
            "raw_density_leftwards": True,
            "raw_density_rightwards": True,
            "raw_speed_proceeding": ":.2f",
            "raw_speed_following": ":.2f",
            "raw_speed_leftwards": ":.2f",
            "raw_speed_rightwards": ":.2f",
            "lat": False,
            "lon": False
        },
        zoom=16,
        height=800,
        title=f"Trajectory Visualization: {input_file} (Click a point to open OSM link)"
    )

    # Add road traces to the figure and move points to the top
    point_traces = list(fig.data)
    fig.add_traces(road_traces)
    
    current_data = list(fig.data)
    n_points = len(point_traces)
    new_data = current_data[n_points:] + current_data[:n_points]
    fig.data = new_data

    fig.update_layout(
        map_style="open-street-map",
        margin={"r":0,"t":40,"l":0,"b":0},
        showlegend=False,
        clickmode="event"
    )

    print("Writing HTML with click-event handler...")
    html_file = "trajectory_map.html"
    base_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    
    js_injection = """
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plotEl = document.getElementsByClassName('plotly-graph-div')[0];
            plotEl.on('plotly_click', function(data){
                if(data.points.length > 0 && data.points[0].customdata) {
                    var url = data.points[0].customdata[0];
                    if(url) {
                        window.open(url, '_blank');
                    }
                }
            });
        });
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
    parser = argparse.ArgumentParser(description="Visualize preprocessed pNEUMA trajectories interactively.")
    parser.add_argument('input', help="Path to the processed CSV file (e.g., processed_data/20181024_dX_0830_0900_processed.csv)")
    args = parser.parse_args()
    visualize(args.input)

if __name__ == '__main__':
    main()
