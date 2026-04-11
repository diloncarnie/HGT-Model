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
    def get_adj_summary(sid, key):
        chain = adjacency.get(str(sid), {}).get(key, [])
        lengths = adjacency.get(str(sid), {}).get(f"{key[:-1]}_lengths", [])
        if not chain:
            return "None"
        total_len = sum(lengths)
        return f"{len(chain)} segments ({total_len:.1f}m): {chain}"

    edges['successors_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'successors'))
    edges['predecessors_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'predecessors'))

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
            "length": True,
            "highway": True,
            "successors_info": True,
            "predecessors_info": True,
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
    js_injection = f"""<div style="padding: 10px; text-align: center; font-family: sans-serif;">
        <input type="text" id="segIdInput" placeholder="Enter Segment ID" style="padding: 5px; width: 200px;">
        <button id="searchButton" style="padding: 5px 10px;">Search & Highlight</button>
    </div>
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

            document.getElementById('searchButton').addEventListener('click', function() {{
                var segId = document.getElementById('segIdInput').value;
                if (segId && geoMap[segId]) {{
                    console.log("Searching for Segment ID:", segId);
                    highlightNeighbors(segId, plotEl);
                    
                    // Also center the map on the found segment
                    let centerCoords = null;
                    for (const p of geoMap[segId]) {{
                        if (p[0] !== null && p[1] !== null) {{
                            centerCoords = p;
                            break;
                        }}
                    }}
                    if (centerCoords) {{
                         Plotly.relayout(plotEl, {{
                            'mapbox.center.lat': centerCoords[0],
                            'mapbox.center.lon': centerCoords[1],
                            'mapbox.zoom': 17
                        }});
                    }}
                }} else {{
                    alert("Segment ID '" + segId + "' not found in map data.");
                }}
            }});

            // Allow pressing Enter in the input field to trigger search
            document.getElementById('segIdInput').addEventListener('keyup', function(event) {{
                if (event.key === "Enter") {{
                    event.preventDefault();
                    document.getElementById('searchButton').click();
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
    parser.add_argument('--network_file', default="osm_network.gpkg", help="Path to osm_network.gpkg")
    parser.add_argument('--adjacency_file', default="topological_adjacency.json", help="Path to topological_adjacency.json")
    args = parser.parse_args()
    
    visualize_adjacency(args.network_file, args.adjacency_file)

if __name__ == '__main__':
    main()