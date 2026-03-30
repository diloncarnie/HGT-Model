# Refactor Request: Integrate Script 1 Logic and Test Functionality into preprocess_pneuma.py

I need to refactor the @preprocess_pneuma.py to incorporate the map-matching, geometry calculation, lane assignment, and testing logic from @extract_map_and_speeds.py while strictly maintaining the second preprocess_pneuma.py's high-level architecture.

### **1. Updated Helper: get_osm_map_and_edges**
Update the `get_osm_map_and_edges` helper function to ensure it returns the geometry `coords` list for every edge. This is critical for the sub-segment iteration in the matching phase. It should store the full list of points $(x, y)$ that define each road segment so they can be accessed during the signed distance calculation.

### **2. Test Flag & Main Loop (main function)**
Refactor the `main()` function to include the following:
* **Test Mode Sampling:** Incorporate the `test_percentage` logic from extract_map_and_speeds.py. Update the argument parser and the file-reading section to handle the `--test` flag. If active, the script must count non-motorcycle vehicles and limit processing to the specified percentage of the total (e.g., 75%), mirroring extract_map_and_speeds.py’s behavior.
* **Integrated Lane Assignment (Post-Merge):** After the chunks are merged into the final dataframe, replace the existing lane assignment with the extract_map_and_speeds.py methodology. Use the **Global Anchor** logic: `df.loc[group.index, 'D'] = (abs_max_d - group['signed_dist']).clip(lower=0.0)`. Then, apply the `np.digitize` and `np.clip` logic to assign a clean `lane_index` based on the pre-loaded `lane_boundaries.json`.

### **3. Map-Matching & Geometry (process_chunk function)**
Refactor the `process_chunk` function to use the extract_map_and_speeds.py methodology while preserving preprocess_pneuma.py's line-by-line reading and multiprocessing pool:
* **Iterative Matching:** Implement the loop `for max_dist in range(5, 25, 5):` to try progressively larger distances until a trajectory is fully matched.
* **Simultaneous Geometry Calculation:** Inside the vehicle loop, iterate through road sub-segments to calculate `best_dist`, `signed_d` (using the cross-product method to determine the side of the road), and `dist_along` (for `t_proj`) in a single pass. 
* **Attribute Filtering:** I do NOT need vehicle azimuth (`azcar`) or relative heading (`rel_heading`). 
* **Data Consistency:** The function must return a list containing: `signed_dist`, `t_proj`, `segment_id`, `x`, and `y`, ensuring all downstream columns required for the final CSV remain available.

### **4. Preservation of Feature Extraction**
The rest of the functionality in preprocess_pneuma.py must remain unchanged:
* **Frenet Ego-Centric Extraction:** The KDTree logic in `process_frenet_for_timestamp` (calculating densities and relative speeds for `proceeding`, `following`, `leftwards`, and `rightwards` zones) must continue to function exactly as written, using the newly calculated `D` and `t_proj` values.
* **Final Output:** The final CSV structure, column names, and kinematics calculations (like `relative_kinematic_ratio`) must remain identical to the original preprocess_pneuma.py output.

Additionally, read @visualize_map_matching.py and create a new visualizer script for the output of preprocess_pneuma.py. In this new script, we do not need to click on the poitns, just to hover over and examine information. Make sure to include the same kind of informiaton when hoverping plus the relative ego centric features such as the relative density and relative speeds in each zone.