We need to change the way in which we are implementing the fundamental functionality of the preprocess_pneuma.py script. Additionally, lets create a extract_map_and_speeds.py script to perform some intial steps to allow the main script to perform 1 pass over each csv file instead of two.

Follow the step-by-step algorithmic instructions provided for script 1:

### Script 1: `extract_map_and_speeds.py` (Initialization)
**Objective:** Process a single pNEUMA CSV to generate the static foundational map, free-flow speed, and topology files.

**Execution Steps & Constraints:**
1. **OSM Network:** Download the `drive` network for the Athens bounding box `(23.70, 37.95, 23.76, 38.00)` using `osmnx`. Filter to retain ONLY `primary`, `secondary`, `tertiary`, and `trunk`. Project to UTM Cartesian (e.g., EPSG:32634). 
2. **Topological Adjacency (CRITICAL):** To solve Frenet boundary blindspots, implement the exact function below to calculate "straight-ahead" connections. Run it on the UTM edges:

```python
import numpy as np
import math

def build_topological_adjacency(edges_gdf):
    """/
    Builds a dictionary mapping every segment_id to its valid, 
    'straight-ahead' successors and 'straight-behind' predecessors.
    """
    # Create quick lookups
    edges_by_u = edges_gdf.groupby('u')['segment_id'].apply(list).to_dict()
    edges_by_v = edges_gdf.groupby('v')['segment_id'].apply(list).to_dict()
    
    adjacency = {}
    
    for _, row in edges_gdf.iterrows():
        ego_id = row['segment_id']
        u_node, v_node = row['u'], row['v']
        ego_geom = row['geometry']
        
        # Calculate Ego Exit Heading (using last two points of LineString)
        ego_coords = list(ego_geom.coords)
        ego_dx = ego_coords[-1][0] - ego_coords[-2][0]
        ego_dy = ego_coords[-1][1] - ego_coords[-2][1]
        ego_heading = math.degrees(math.atan2(ego_dy, ego_dx))
        
        valid_successors = []
        
        # Check all branches leaving the end node (v)
        potential_successors = edges_by_u.get(v_node, [])
        for succ_id in potential_successors:
            if succ_id == ego_id: continue
            
            succ_geom = edges_gdf[edges_gdf['segment_id'] == succ_id].iloc[0]['geometry']
            succ_coords = list(succ_geom.coords)
            
            # Calculate Successor Entry Heading (using first two points)
            succ_dx = succ_coords[1][0] - succ_coords[0][0]
            succ_dy = succ_coords[1][1] - succ_coords[0][1]
            succ_heading = math.degrees(math.atan2(succ_dy, succ_dx))
            
            # Calculate Deflection Angle
            delta_theta = (succ_heading - ego_heading + 180) % 360 - 180
            
            # Filter: Only keep branches within a 45-degree straight cone
            if abs(delta_theta) <= 45.0:
                valid_successors.append(succ_id)
                
        adjacency[ego_id] = {
            'successors': valid_successors,
            # (Similar logic would be applied in reverse for predecessors)
            'predecessors': [] 
        }
        
    return adjacency
```

3. **Map-Matching (Iterative HMM):** Parse the single CSV. Filter for "Car" only. Map-match to the UTM network using the `leuvenmapmatching.matcher.distance` module. 
    * **The Iterative Loop:** Initialize the matcher with `max_dist=5`, `max_dist_init=50`, `min_prob_norm=0.5`, `non_emitting_length_factor=0.75`, `obs_noise=1`.
    * Wrap the `matcher.match(trace)` call in a `try-except` block catching `NoMatchError`.
    * If it fails, increment `max_dist` by 5 meters and retry. Cap the retry loop at a maximum `max_dist` of 50 meters before dropping the trajectory completely.
4. **Empirical Speeds:** Calculate the traversal speed for every matched vehicle. For every `segment_id`, calculate the 95th percentile speed (m/s). Impute missing segments using highway type averages: primary=14, secondary=11, tertiary=9, trunk=20.
5. **Exports:** Export `osm_network.gpkg` (convert list-like tags to strings), `empirical_free_flow_speeds.json`, and `topological_adjacency.json`.

---

Now follow the the changes that need to be made for the preprocess_pneuma.py script for the three aspects of functionality: Map-Matching, Lane Detection and ego-centric feature extraction. Modify the relevant functions in the original script, and maintain the initial instructions such as the relevant features to calculate and exact columns to create in the final preprocessed csv file for each csv file to be preprocessed.

### Script 2: `preprocess_pneuma.py` (Main Pipeline)
**Objective:** A Single-Pass pipeline processing all CSVs utilizing the static files.

**Execution Steps & Constraints:**
1. **Initialization:** Load the GPKG, free-flow speeds JSON, and adjacency JSON into memory. You will use the free-flow speeds for every segment to calcualte the relative_feature columns to allow for one pass for each csv file instead of the previous two-pass architecutre.
2. **Unified Map-Matching & Kinematic Filtering:** Map-match ALL vehicle types using the exact Iterative HMM Loop defined in Script 1. Apply these two post-match filters:
    * **Bearing Filter:** Calculate the vehicle's trajectory vector (first point to last point) and the matched edge's geometric vector. Calculate the angular difference. Drop the trajectory if $ \Delta \theta > 90^\circ $.
    * **Distance Filter:** Calculate the absolute orthogonal distance of every point to the matched edge. Drop the trajectory if $ \max(distance) > 10m $.
3. **Data-Driven Lane Detection (Continuous Lateral Framing):** Abandon static OSM lane offsets. Implement this exact matrix sequence:
    * **Step A (Signed Distance):** For EVERY vehicle point $ P $, find the closest sub-segment $ A \to B $ on the matched OSM edge. Calculate the 2D cross product: $ \text{Cross} = (B_x - A_x)(P_y - A_y) - (B_y - A_y)(P_x - A_x) $. Assign a negative sign to the absolute distance if Cross > 0 (Left), positive if Cross < 0 (Right).
    * **Step B (Anchor to Edge):** For each unique `segment_id`, find the minimum signed distance across all vehicles. Subtract this minimum from all signed distances on that segment. Store this as the $ D $ coordinate (`distance_from_right_edge`).
    * **Step C (Jenks Loop):** Filter data to "Car/Taxi" only. Sub-sample a maximum of 5,000 $ D $ coordinates per segment. Initialize $ n = \text{num\_lanes} $. Pass to `jenkspy.jenks_breaks(data, n_classes=n)`. 
    * **Step D (Width Constraint):** Calculate the difference between consecutive breaks. If any difference $ > 4.0m $, increment $ n = n + 1 $ and re-run `jenks_breaks`. Repeat until all bucket widths are $ \le 4.0m $.
    * **Step E (Assignment):** Use `np.digitize` with the final optimized breaks to assign the `lane_index` to ALL vehicle types.
4. **Cross-Segment Frenet Feature Extraction:** Use a Frenet-Serret evaluation to extract surrounding neighbors for ego-centric features. 
   * **Coordinates:** $ S = \text{raw\_offset} $ (longitudinal). $ D = \text{distance\_from\_right\_edge} $ (lateral).
   * **Adjacency Lookup:** For an Ego CAV on `segment_id`, query the KD-Tree but explicitly filter neighbors to include ONLY those whose `segment_id` matches the Ego's `segment_id`, OR exists in the Ego's `successors` list, OR exists in the Ego's `predecessors` list. Apply a directional dot product filter ($>0$) to drop oncoming traffic.
   * **Relative Kinematic Math:** Calculate $ \Delta D = D_{neighbor} - D_{ego} $. Calculate $ \Delta S $ conditionally based on the topological relationship:
      * If Same Segment: $ \Delta S = S_{neighbor} - S_{ego} $
      * If Successor (Downstream): $ \Delta S = (Length_{ego} - S_{ego}) + S_{neighbor} $
      * If Predecessor (Upstream): $ \Delta S = -(S_{ego} + (Length_{neighbor} - S_{neighbor})) $
   * **Frenet Bounding Boxes:** Sort into 4 zones based strictly on these curvilinear bounds:
      * Proceeding: $ 0 < \Delta S \le 50 $ AND $ -1.6 \le \Delta D \le 1.6 $
      * Following: $ -50 \le \Delta S < 0 $ AND $ -1.6 \le \Delta D \le 1.6 $
      * Leftwards: $ -50 \le \Delta S \le 50 $ AND $ \Delta D > 1.6 $
      * Rightwards: $ -50 \le \Delta S \le 50 $ AND $ \Delta D < -1.6 $
   * Compute raw density (count) and average speed (m/s) for captured vehicles. Impute speed to 1.0 (relative) if a box is empty.
5. **Deferred Kinematics:** Drop all `is_CAV == False` rows. Calculate `change_in_euclidean_distance`, `relative_kinematic_ratio`, and `relative_ego_speed` exclusively on the remaining CAVs using the JSON speeds for normalization to save overhead.

First create a detailed plan on how to implement these refactored changes to the existing script that you have developed. After I have approved this plan, execute the plan in the current directory.