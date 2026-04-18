import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import json
import numpy as np
import re

def visualize_adjacency(network_file, adjacency_file):
    if not os.path.exists(network_file):
        print(f"Error: Network file '{network_file}' not found.")
        return
        
    if not os.path.exists(adjacency_file):
        print(f"Error: Adjacency file '{adjacency_file}' not found.")
        return

    print(f"Loading data...")
    edges = gpd.read_file(network_file)
    
    # Calculate filled junction shapes using original CRS (UTM) for accurate metrics
    from shapely.ops import polygonize
    junction_edges_utm = edges[edges['is_internal_junction'].astype(str).str.lower() == 'true']

    # Deduplicate by undirected endpoint pair: two-way roads produce two directed
    # edges with overlapping reversed geometry, which blocks polygonize from closing
    # shapes that include a two-way segment.
    seen_pairs = set()
    unique_junction_geoms = []
    for u, v, geom in zip(junction_edges_utm['u'], junction_edges_utm['v'], junction_edges_utm.geometry):
        key = frozenset((u, v))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_junction_geoms.append(geom)

    polygons_utm = list(polygonize(unique_junction_geoms))
    
    poly_stats = []
    if polygons_utm:
        for poly in polygons_utm:
            area_sqm = poly.area
            perimeter_m = poly.length
            
            bound = poly.boundary
            edge_count = 0
            for edge_geom in junction_edges_utm.geometry:
                if edge_geom.intersection(bound).length > 1e-5:
                    edge_count += 1
                    
            poly_stats.append({
                'area': area_sqm,
                'perimeter': perimeter_m,
                'edge_count': edge_count
            })
            
        polys_gdf = gpd.GeoDataFrame(geometry=polygons_utm, crs=edges.crs)
        polys_gdf_4326 = polys_gdf.to_crs("EPSG:4326")
        polygons_4326 = polys_gdf_4326.geometry.tolist()
    else:
        polygons_4326 = []
        
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

    edges['to_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'to'))
    edges['from_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'from'))
    edges['crossed_by_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'crossed_by'))
    edges['turns_into_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'turns_into'))
    edges['merges_into_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'merges_into'))
    edges['u_turns_into_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'u_turns_into'))
    edges['crosses_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'crosses'))
    edges['opposite_direction_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'opposite_direction'))
    edges['banned_successors_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'banned_successors'))
    edges['only_successors_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'only_successors'))
    edges['merges_with_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'merges_with'))
    edges['merged_by_info'] = edges['segment_id_str'].apply(lambda x: get_adj_summary(x, 'merged_by'))
    
    # Ensure relevant columns exist before adding them to hover data
    for col in ['turn:lanes', 'turn', 'junction', 'is_internal_junction', 'is_shape_junction', 'is_semantic_junction']:
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
    osmid_map = {}
    
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
        
        # Safely extract all numeric OSM Way IDs (in case OSMnx merged them into a string list like "[123, 456]")
        osmid_raw = str(row.get('osmid', ''))
        matches = re.findall(r'\d+', osmid_raw)
        osmid_map[row['segment_id_str']] = matches if matches else []

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
            "lanes": True,
            "turn:lanes": True,
            "junction": True,
            "is_internal_junction": True,
            "is_shape_junction": True,
            "is_semantic_junction": True,
            "to_info": True,
            "from_info": True,
            "crossed_by_info": True,
            "turns_into_info": True,
            "merges_into_info": True,
            "u_turns_into_info": True,
            "crosses_info": True,
            "opposite_direction_info": True,
            "banned_successors_info": True,
            "only_successors_info": True,
            "merges_with_info": True,
            "merged_by_info": True,
            "lat": False,
            "lon": False
        },
        zoom=15,
        height=800,
        title="Adjacency Debugger: Click a segment point to highlight its neighbors"
    )

    # Add background roads, junction highlighted roads, and filled junction shapes
    bg_lats, bg_lons = [], []
    junc_lats, junc_lons = [], []
    
    for idx, row in edges.iterrows():
        sid = row['segment_id_str']
        coords = geo_map.get(sid, [])
        is_junc = str(row.get('is_internal_junction', 'False')).lower() == 'true'
        
        for pt in coords:
            if pt[0] is not None:
                if is_junc:
                    junc_lats.append(pt[0])
                    junc_lons.append(pt[1])
                else:
                    bg_lats.append(pt[0])
                    bg_lons.append(pt[1])
        
        if is_junc:
            junc_lats.append(None)
            junc_lons.append(None)
        else:
            bg_lats.append(None)
            bg_lons.append(None)
            
    # Prepare filled junction shapes traces using GeoJSON for face hover
    shape_features = []
    shape_locations = []
    shape_z = []
    shape_hovertexts = []
    
    for i, poly in enumerate(polygons_4326):
        stats = poly_stats[i]
        hover_text = f"<b>Junction Shape</b><br>Area: {stats['area']:.1f} sq m<br>Perimeter: {stats['perimeter']:.1f} m<br>Edges: {stats['edge_count']}"
        
        shape_locations.append(str(i))
        shape_z.append(1)
        shape_hovertexts.append(hover_text)
        
        shape_features.append({
            "type": "Feature",
            "id": str(i),
            "geometry": poly.__geo_interface__,
            "properties": {}
        })

    geojson_data = {
        "type": "FeatureCollection",
        "features": shape_features
    }
    
    bg_roads = go.Scattermap(
        lat=bg_lats,
        lon=bg_lons,
        mode='lines',
        line=dict(width=2, color='rgba(100, 100, 100, 0.2)'),
        hoverinfo='skip',
        showlegend=False
    )
    
    junc_roads = go.Scattermap(
        lat=junc_lats,
        lon=junc_lons,
        mode='lines',
        line=dict(width=3, color='rgba(255, 165, 0, 0.8)'),
        name='Internal Junction Links',
        hoverinfo='skip',
        showlegend=True
    )
    
    # Use Choroplethmap to allow hovering over the polygon faces
    junc_shapes = go.Choroplethmap(
        geojson=geojson_data,
        locations=shape_locations,
        z=shape_z,
        colorscale=[[0, 'rgba(255, 0, 0, 0.3)'], [1, 'rgba(255, 0, 0, 0.3)']],
        marker_line_width=1,
        marker_line_color='rgba(255, 0, 0, 0.8)',
        showscale=False,
        text=shape_hovertexts,
        hoverinfo='text',
        name='Junction Shapes'
    )
    
    fig.add_trace(bg_roads)
    fig.add_trace(junc_roads)
    fig.add_trace(junc_shapes)
    
    # Reorder: background roads at bottom, then junction shapes, then junction lines, then scatter points
    data_list = list(fig.data)
    fig.data = (data_list[-3], data_list[-1], data_list[-2], data_list[0])

    print("Preparing interactive HTML...")

    html_file = "adjacency_debugger.html"
    base_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    
    # Fix JS injection with correct syntax and robust data lookup
    js_injection = f"""<div style="padding: 10px; text-align: center; font-family: sans-serif;">
        <input type="text" id="segIdInput" placeholder="Enter Segment ID" style="padding: 5px; width: 200px;">
        <button id="searchButton" style="padding: 5px 10px;">Search & Highlight</button>
        <span id="osmLinkContainer" style="margin-left: 20px; font-size: 15px;"></span>
    </div>
    <script>
        var adjacency = {json.dumps(adjacency)};
        var geoMap = {json.dumps(geo_map)};
        var osmidMap = {json.dumps(osmid_map)};
        
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
            var adj = adjacency[egoIdStr] || {{to: [], from: [], merges_into: [], crossed_by: [], turns_into: [], u_turns_into: [], crosses: [], opposite_direction: [], banned_successors: [], only_successors: [], merges_with: [], merged_by: []}};
            
            // Update OSM Links in the control bar
            var osmids = osmidMap[egoIdStr];
            var linkContainer = document.getElementById('osmLinkContainer');
            if (osmids && osmids.length > 0 && linkContainer) {{
                var linksHTML = '🌍 View Segment ' + egoIdStr + ': ';
                var linkArray = osmids.map(function(id) {{
                    return '<a href="https://www.openstreetmap.org/way/' + id + '" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;">OSM Way ' + id + '</a>';
                }});
                linkContainer.innerHTML = linksHTML + linkArray.join(' | ');
            }} else if (linkContainer) {{
                linkContainer.innerHTML = '';
            }}
            
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
            adj.to.forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 6, color: 'lime'}},
                        name: egoIdStr + ' to ' + sid
                    }});
                }}
            }});
            
            // 3. Predecessors (Red)
            adj.from.forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 6, color: 'red'}},
                        name: egoIdStr + ' from ' + sid
                    }});
                }}
            }});

            // 4. Intersects In (Cyan) - Segments flowing INTO the ego segment intersection
            (adj.crossed_by || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 5, color: 'cyan'}},
                        name: egoIdStr + ' crossed_by ' + sid
                    }});
                }}
            }});

            // 5. Intersects Out (Magenta) - Segments flowing OUT OF the ego segment intersection
            (adj.turns_into || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 5, color: 'magenta'}},
                        name: egoIdStr + ' turns_into ' + sid
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

            // 8. Merging Into (Orange)
            (adj.merges_into || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 7, color: 'orange'}},
                        name: egoIdStr + ' merges_into ' + sid
                    }});
                }}
            }});

            // 9. U-Turns (Brown)
            (adj.u_turns_into || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 6, color: 'brown'}},
                        name: egoIdStr + ' u-turns_into ' + sid
                    }});
                }}
            }});

            // 10. Crosses (Teal)
            (adj.crosses || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 6, color: 'teal'}},
                        name: egoIdStr + ' crosses ' + sid
                    }});
                }}
            }});

            // 11. Merges With (Yellow)
            (adj.merges_with || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 6, color: 'yellow'}},
                        name: egoIdStr + ' merges_with ' + sid
                    }});
                }}
            }});

            // 12. Merged By (Pink)
            (adj.merged_by || []).forEach(sid => {{
                if (geoMap[sid]) {{
                    traces.push({{
                        type: 'scattermap',
                        lat: geoMap[sid].map(p => p[0]),
                        lon: geoMap[sid].map(p => p[1]),
                        mode: 'lines',
                        line: {{width: 6, color: 'pink'}},
                        name: egoIdStr + ' merged_by ' + sid
                    }});
                }}
            }});

            // Clear previous highlight traces (indices 4 onwards)
            var currentTracesCount = plotEl.data.length;
            var indicesToRemove = [];
            for (var i = 4; i < currentTracesCount; i++) indicesToRemove.push(i);

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
    parser.add_argument('--network-file', default="osm_network.gpkg", help="Path to osm_network.gpkg")
    parser.add_argument('--adjacency_file', default="topological_adjacency.json", help="Path to topological_adjacency.json")
    args = parser.parse_args()
    
    visualize_adjacency(args.network_file, args.adjacency_file)

if __name__ == '__main__':
    main()