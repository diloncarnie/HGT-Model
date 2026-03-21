import argparse
import pandas as pd
import plotly.express as px
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

    osmid_map = {}
    for idx, row in edges.iterrows():
        way_id_str = str(row['osmid'])
        # Handle string representations of lists from the GPKG export
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

    # Ensure track_id is categorical for discrete color mapping
    df['track_id_str'] = df['track_id'].astype(str)

    print("Generating interactive Map visualization...")
    fig = px.scatter_map(
        df,
        lat="lat",
        lon="lon",
        color="track_id_str",
        custom_data=["osm_url"], # Store raw URL for click events
        hover_data={
            "track_id_str": False, # Hide duplicate
            "track_id": True,
            "speed": ":.2f", # Format to 2 decimal places
            "segment_id": True,
            "osm_url": False, # Hide raw URL from hover
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

    fig.update_layout(
        map_style="open-street-map",
        margin={"r":0,"t":40,"l":0,"b":0},
        showlegend=False,
        clickmode="event" # Enable click events
    )

    print("Writing HTML with click-event handler...")
    html_file = "trajectory_map.html"
    
    # Generate the base HTML
    base_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    
    # Inject JavaScript to handle point clicks by looking up the custom_data URL
    js_injection = """
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plotEl = document.getElementsByClassName('plotly-graph-div')[0];
            plotEl.on('plotly_click', function(data){
                if(data.points.length > 0) {
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
    
    # Replace the closing tags with our script
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
