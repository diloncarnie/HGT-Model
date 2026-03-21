import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import json

def visualize_adjacency(network_file, adjacency_file):
    print(f"Loading data...")
    edges = gpd.read_file(network_file)
    if edges.crs != "EPSG:4326":
        edges = edges.to_crs("EPSG:4326")
        
    with open(adjacency_file, 'r') as f:
        adjacency = json.load(f)

    edges['segment_id_str'] = edges['segment_id'].astype(str)
    
    # Add adjacency info to edges for hover
    def get_adj_str(sid, key):
        items = adjacency.get(str(sid), {}).get(key, [])
        if len(items) > 5:
            return f"{len(items)} items: {items[:5]}..."
        return str(items)

    edges['successors_list'] = edges['segment_id_str'].apply(lambda x: get_adj_str(x, 'successors'))
    edges['predecessors_list'] = edges['segment_id_str'].apply(lambda x: get_adj_str(x, 'predecessors'))

    print("Generating base map...")
    # Hack: use scatter_map on the centroids for easy clicking
    edges['centroid'] = edges.geometry.centroid
    edges['lat'] = edges['centroid'].y
    edges['lon'] = edges['centroid'].x

    fig = px.scatter_map(
        edges,
        lat="lat",
        lon="lon",
        custom_data=["segment_id_str"], # Explicitly set for JS retrieval
        hover_data={
            "segment_id": True,
            "successors_list": True,
            "predecessors_list": True,
            "lat": False,
            "lon": False
        },
        zoom=15,
        height=800,
        title="Adjacency Debugger: Click a segment point to highlight its neighbors"
    )

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
    fig.data = (data_list[1], data_list[0])

    print("Preparing interactive HTML...")
    
    # Map of segment_id -> coords for JS
    geo_map = {}
    for _, row in edges.iterrows():
        coords = []
        if row.geometry.geom_type == 'LineString':
            coords = [[y, x] for x, y in zip(*row.geometry.xy)]
        elif row.geometry.geom_type == 'MultiLineString':
            # Simplified for highlighting: just take first part or flatten
            for line in row.geometry.geoms:
                coords.extend([[y, x] for x, y in zip(*line.xy)])
                coords.append([None, None])
        geo_map[row['segment_id_str']] = coords

    html_file = "adjacency_debugger.html"
    base_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    
    # Fix JS injection with correct syntax and robust data lookup
    js_injection = f"""
    <script>
        var adjacency = {json.dumps(adjacency)};
        var geoMap = {json.dumps(geo_map)};
        
        document.addEventListener('DOMContentLoaded', function() {{
            var plotEl = document.getElementsByClassName('plotly-graph-div')[0];
            
            plotEl.on('plotly_click', function(data){{
                if(data.points.length > 0) {{
                    // retrieve from customdata[0] which we set to segment_id_str
                    var segId = data.points[0].customdata[0];
                    console.log("Clicked Segment ID:", segId);
                    highlightNeighbors(segId, plotEl);
                }}
            }});
        }});

        function highlightNeighbors(egoId, plotEl) {{
            var egoIdStr = String(egoId);
            var adj = adjacency[egoIdStr] || {{successors: [], predecessors: []}};
            
            var traces = [];
            
            // 1. Ego (Gold)
            if (geoMap[egoIdStr]) {{
                traces.push({{
                    type: 'scattermap',
                    lat: geoMap[egoIdStr].map(p => p[0]),
                    lon: geoMap[egoIdStr].map(p => p[1]),
                    mode: 'lines',
                    line: {{width: 8, color: 'gold'}},
                    name: 'Selected: ' + egoIdStr
                }});
            }}
            
            // 2. Successors (Green)
            adj.successors.forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 6, color: 'lime'}},
                        name: 'Successor: ' + sid
                    }});
                }}
            }});
            
            // 3. Predecessors (Red)
            adj.predecessors.forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 6, color: 'red'}},
                        name: 'Predecessor: ' + sid
                    }});
                }}
            }});

            // Clear previous highlight traces (indices 2 onwards)
            var currentTracesCount = plotEl.data.length;
            var indicesToRemove = [];
            for (var i = 2; i < currentTracesCount; i++) indicesToRemove.push(i);
            
            // Note: deleteTraces is synchronous but indices shift. 
            // It's safer to remove from end or use a single call.
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
    parser = argparse.ArgumentParser(description="Interactively visualize road adjacency.")
    parser.add_argument('network_file', help="Path to osm_network.gpkg")
    parser.add_argument('adjacency_file', help="Path to topological_adjacency.json")
    args = parser.parse_args()
    
    visualize_adjacency(args.network_file, args.adjacency_file)

if __name__ == '__main__':
    main()