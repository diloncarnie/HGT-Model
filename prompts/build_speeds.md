
Please make a plan to create two completely separate Python scripts: `build_speed_graph.py` and `visualize_speeds.py`. After my approval, proceed to craete these scripts in the current directory.

### Script 1: `build_speed_graph.py`
**Objective:** Download the OpenStreetMap network, hit the Google Routes API to find static free-flow speeds, and export the network and a JSON speed dictionary.

**Execution Steps & Constraints:**
1. **OSM Network:** Use `osmnx` to download the `drive` network for the Athens bounding box `(23.70, 37.95, 23.76, 38.00)`. Filter the graph to retain ONLY `primary`, `secondary`, `tertiary`, and `trunk` highways. 
2. **Export Network:** Save the edges to a GeoPackage (`osm_network.gpkg`). *CRITICAL: GPKG cannot save Python lists. You must convert any list-like attributes (like `osmid` or `highway`) to strings before saving.*
3. **The Quantization Filter:** Calculate the length of every edge in meters. Filter out all edges shorter than 50 meters. (Google's duration API returns integers; short edges result in massive quantization errors when calculating speed).
4. **Google Routes API:** For the edges > 50m, use a `ThreadPoolExecutor` to asynchronously POST to `https://routes.googleapis.com/directions/v2:computeRoutes`. 
   - Require a `Maps_API_KEY` environment variable.
   - Payload must use: `"travelMode": "DRIVE"`, `"routingPreference": "TRAFFIC_UNAWARE"`.
   - FieldMask: `routes.duration,routes.distanceMeters`.
   - Calculate speed (m/s) = `distance / duration`.
5. **Fallback Imputation:** For the short edges (< 50m) or API failures, calculate the average retrieved speed for each `highway` type, and assign those averages to the missing segments.
6. **Export JSON:** Save a dictionary mapping `{segment_id: speed_in_ms}` to `google_free_flow_speeds.json`.

---

### Script 2: `visualize_speeds.py`
**Objective:** Load the exported GPKG and JSON, and generate an interactive Plotly HTML map visualizing the free-flow speeds, with clickable segments that route to OpenStreetMap.

**Execution Steps & Constraints:**
1. **Load Data:** Use `argparse` to accept a directory. Load `osm_network.gpkg` and `google_free_flow_speeds.json`. Merge the speeds onto the GeoDataFrame using `segment_id`. Convert speeds to km/h for readability.
2. **OSM URL Generation:** Create a column with the URL `https://www.openstreetmap.org/way/{osmid}`. Handle the stringified list formats that were saved in the GPKG.
3. **Unroll Geometries:** `px.line_map` (or `px.line_mapbox`) requires flat coordinates. Iterate through the `LineString` geometries and unroll them into a flat DataFrame containing `lat`, `lon`, `segment_id`, `Speed (km/h)`, `Road Type`, and `osm_url`.
4. **Plotly Configuration:** Generate the map using `Speed (km/h)` as the color scale. Include the `osm_url` in the `custom_data`. Hide the raw URL and raw coordinates from the hover tooltip, but show the Segment ID, Road Type, and Speeds. Use `map_style="open-street-map"`.
5. **Interactive Click Injection:** Save the figure to `speed_map.html`. You MUST inject JavaScript before the `</body>` tag to listen for Plotly click events (`plotly_click`). When a segment is clicked, the JS should extract the `osm_url` from the `customdata` and open it in a new browser tab.
6. **Auto-Open:** Use the `webbrowser` module to automatically open the HTML file upon completion.

Note that my API key is AIzaSyAUDFIgOrbJeK4uWvZ7CLAnJHCPF7I4-DQ