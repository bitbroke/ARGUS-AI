import json
notebook_path = r'C:\Users\M S I\Downloads\cancer_dataset\Untitled-1.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the validation cell and memory-optimize it
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Model Validation & Performance Study' in source:
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if 'Y_test = Y_val[split_idx:]' in line:
                    new_source.append('\n    # --- MEMORY FIX: Clear large arrays ---\n')
                    new_source.append('    del X_val\n')
                    new_source.append('    del Y_val\n')
                    new_source.append('    del X_all\n')
                    new_source.append('    import gc\n')
                    new_source.append('    gc.collect()\n')
                if 'predicted_congestion = (Y_pred < congestion_threshold * base_future_test).astype(int)' in line:
                    new_source.append('\n    # --- MEMORY FIX: Clear large arrays ---\n')
                    new_source.append('    del base_future_test\n')
                    new_source.append('    gc.collect()\n')
            cell['source'] = new_source

# Append the UI export cell at the end
export_code = '''import tensorflow as tf
import numpy as np
import json
import gc

# 1. Load the saved model
print("Loading Nexus Flow model...")
nexus_model = tf.keras.models.load_model('backend/data/nexus_flow_model.keras')

# 2. Select a single, high-impact traffic window for the Demo
# Taking just the first sequence from your test set to save memory
demo_X = X_test[0:1]  # Shape: (1, 12, 2290, 3)
demo_Y_actual = Y_test[0:1]

# 3. Run Inference
print("Generating 3-hour future simulation...")
demo_predictions = nexus_model.predict(demo_X) # Shape: (1, 3, 2290, 1)

# Clear RAM
del X_test
del Y_test
try:
    del Y_pred
except:
    pass
gc.collect()

# 4. Apply Dynamic Thresholding & Format for Mapbox UI
print("Applying Dynamic Thresholds and formatting payload...")
ui_payload = {
    "simulation_hours": 3,
    "nodes": []
}

# Iterate through nodes to build the JSON
for node_idx in range(num_nodes):
    node_id = list(G.nodes())[node_idx]
    degree = G.degree(node_id)
    
    # Apply Fix #1 Logic
    if degree <= 2:
        threshold_multiplier = 0.75 # 25% drop
    elif degree == 3:
        threshold_multiplier = 0.85 # 15% drop
    else:
        threshold_multiplier = 0.95 # 5% drop (Hyper-sensitive)
        
    baseline = demo_X[0, -1, node_idx, 0] # Traffic volume at Hour 0
    
    node_data = {
        "node_id": str(node_id),
        "lat": float(G.nodes[node_id]['y']),
        "lon": float(G.nodes[node_id]['x']),
        "degree": degree,
        "predictions": []
    }
    
    for future_hour in range(3):
        pred_vol = float(demo_predictions[0, future_hour, node_idx, 0])
        is_congested = bool(pred_vol < (baseline * threshold_multiplier))
        
        node_data["predictions"].append({
            "hour": future_hour + 1,
            "predicted_volume": max(0, round(pred_vol, 2)),
            "congested": is_congested
        })
        
    ui_payload["nodes"].append(node_data)

# 5. Export to JSON
export_path = 'ui_traffic_simulation.json'
with open(export_path, 'w') as f:
    json.dump(ui_payload, f)

print(f"Success! UI Payload saved to {export_path}")
'''

# Check if the export cell is already there, if not append it
if not any('ui_traffic_simulation.json' in ''.join(c['source']) for c in nb['cells']):
    nb['cells'].append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [line + '\n' for line in export_code.split('\n')]
    })

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook updated successfully!')
