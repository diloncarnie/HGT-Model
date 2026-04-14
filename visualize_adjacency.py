import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import json
import numpy as np

def visualize_adjacency(network_file, adjacency_file):
    if not os.path.exists(network_file):
        print(f"Error: Network file '{network_file}' not found.")
        return
        
    if not os.path.exists(adjacency_file):
        print(f"Error: Adjacency file '{adjacency_file}' not found.")
        return

    print(f"Loading data...")
    edges = gpd.read_file(network_file)
    if edges.crs != "EPSG:4326":
        edges = edges.to_crs("EPSG:4326")
        
    with open(adjacency_file, 'r') as f:
        adjacency = json.load(f)

    edges['segment_id_str'] = edges['segment_id'].astype(str)
    
    # Add adjacency info to edges for hover
    def get_adj_summary(sid, key):
        node_data = adjacency.get(str(sid), {})
        chain = node_data.get(key, [])
        if not chain:
            return "None"
        length_key = f"{key[:-1]}_lengths" if key.endswith('s') else f"{key}_lengths"
        lengths = node_data.get(length_key, [])
        if lengths and len(lengths) == len(chain):
            total_len = sum(lengths)
            return f"{len(chain)} segments ({total_len:.1f}m): {chain}"
        return f"{len(chain)} segments: {chain}"

    edges['successors_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'successors'))
    edges['predecessors_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'predecessors'))
    edges['intersects_in_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'intersects_in'))
    edges['intersects_out_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'intersects_out'))
    edges['opposite_direction_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'opposite_direction'))
    edges['banned_successors_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'banned_successors'))
    edges['only_successors_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'only_successors'))
    
    # Ensure turn information columns exist before adding them to hover data
    for col in ['turn:lanes', 'turn']:
        if col not in edges.columns:
            edges[col] = "None"
        else:
            edges[col] = edges[col].fillna("None").astype(str)

    print("Generating base map...")
    # Calculate rightward perpendicular shift to separate perfectly overlapping opposite-direction segments
    SHIFT_OFFSET = 0.00003 # ~3 meters offset
    
    lats = []
    lons = []
    geo_map = {}
    
    for idx, row in edges.iterrows():
        geom = row.geometry
        
        # Calculate overall direction vector of the segment
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            coords = list(geom.geoms[0].coords)
        else:
            coords = []
            
        if len(coords) >= 2:
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            length = np.hypot(dx, dy)
            if length > 0:
                nx = dy / length
                ny = -dx / length
            else:
                nx, ny = 0.0, 0.0
        else:
            nx, ny = 0.0, 0.0
            
        # Check if the segment is a one-way street
        is_oneway = str(row.get('oneway', 'False')).lower() in ['true', 'yes', '1']
        
        if is_oneway:
            shift_lon = 0.0
            shift_lat = 0.0
        else:
            shift_lon = nx * SHIFT_OFFSET
            shift_lat = ny * SHIFT_OFFSET
        
        centroid = geom.centroid
        lats.append(centroid.y + shift_lat)
        lons.append(centroid.x + shift_lon)
        
        # Shift the visualized geometry rightward as well for the frontend geo_map
        shifted_coords = []
        if geom.geom_type == 'LineString':
            shifted_coords = [[y + shift_lat, x + shift_lon] for x, y in zip(*geom.xy)]
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                shifted_coords.extend([[y + shift_lat, x + shift_lon] for x, y in zip(*line.xy)])
                shifted_coords.append([None, None])
                
        geo_map[row['segment_id_str']] = shifted_coords

    edges['lat'] = lats
    edges['lon'] = lons

    fig = px.scatter_map(
        edges,
        lat="lat",
        lon="lon",
        custom_data=["segment_id_str"], # Explicitly set for JS retrieval
        hover_data={
            "segment_id": True,
            "length": True,
            "highway": True,
            "oneway": True,
            "turn:lanes": True,
            "turn": True,
            "successors_info": True,
            "predecessors_info": True,
            "intersects_in_info": True,
            "intersects_out_info": True,
            "banned_successors_info": True,
            "only_successors_info": True,
            "opposite_direction_info": True,
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
    for coords in geo_map.values():
        for pt in coords:
            lats.append(pt[0])
            lons.append(pt[1])
        lats.append(None)
        lons.append(None)
    
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
            var adj = adjacency[egoIdStr] || {{successors: [], predecessors: [], intersects_in: [], intersects_out: [], opposite_direction: [], banned_successors: [], only_successors: []}};
            
            var traces = [];
            
            // 1. Ego (Gold)
            if (geoMap[egoIdStr]) {{
                var coords = geoMap[egoIdStr];
                traces.push({{
                    type: 'scattermap',
                    lat: coords.map(p => p[0]),
                    lon: coords.map(p => p[1]),
                    mode: 'lines',
                    line: {{width: 8, color: 'gold'}},
                    name: 'Selected: ' + egoIdStr
                }});
                
                // Find the first valid point to mark as START
                var firstValid = null;
                for (var i = 0; i < coords.length; i++) {{
                    if (coords[i][0] !== null) {{ firstValid = coords[i]; break; }}
                }}
                
                // Find the last valid point to mark as END
                var lastValid = null;
                for (var i = coords.length - 1; i >= 0; i--) {{
                    if (coords[i][0] !== null) {{ lastValid = coords[i]; break; }}
                }}
                
                if (firstValid && lastValid) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: [firstValid[0], lastValid[0]],
                        lon: [firstValid[1], lastValid[1]],
                        mode: 'markers+text',
                        marker: {{size: [10, 14], color: ['white', 'black']}},
                        text: ['START', 'END'],
                        textposition: 'top center',
                        textfont: {{size: 14, color: 'black'}},
                        name: 'Direction'
                    }});
                }}
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

            // 4. Intersects In (Cyan) - Segments flowing INTO the ego segment intersection
            (adj.intersects_in || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 5, color: 'cyan'}},
                        name: 'Intersects (In): ' + sid
                    }});
                }}
            }});

            // 5. Intersects Out (Magenta) - Segments flowing OUT OF the ego segment intersection
            (adj.intersects_out || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 5, color: 'magenta'}},
                        name: 'Intersects (Out): ' + sid
                    }});
                }}
            }});

            // 6. Opposite Direction (Purple) - Overlapping opposite segment
            (adj.opposite_direction || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 5, color: 'purple'}},
                        name: 'Opposite Dir: ' + sid,
                        visible: 'legendonly' // Hides it on the map, keeps it in the legend!
                    }});
                }}
            }});

            // 7. Banned Successors (Black)
            (adj.banned_successors || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 4, color: 'black'}},
                        name: 'Banned Turn: ' + sid
                    }});
                }}
            }});

            // 8. Only Successors (Blue)
            (adj.only_successors || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 6, color: 'blue'}},
                        name: 'Mandatory Turn: ' + sid
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