# -*- coding: utf-8 -*-
"""
Argus AI - Comprehensive Stress Test Script
Tests all backend ML models, API endpoints, graph integrity, and edge cases.
"""
import urllib.request
import json
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor

# Fix Windows console encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_URL = "http://localhost:8000"

def make_request(method, path, body=None, timeout=30):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        elapsed = time.time() - t0
        payload = json.loads(resp.read().decode())
        return resp.status, elapsed, payload, None
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        try:
            err_payload = json.loads(e.read().decode())
        except:
            err_payload = None
        return e.code, elapsed, err_payload, str(e)
    except Exception as e:
        elapsed = time.time() - t0
        return 0, elapsed, None, str(e)

def print_result(name, status, elapsed, error=None, details=""):
    if error is None and status == 200:
        print(f"  [PASS] {name} ({elapsed:.2f}s) {details}")
    else:
        print(f"  [FAIL] {name} ({elapsed:.2f}s) Status: {status}, Error: {error} {details}")

print("=" * 60)
print("ARGUS AI - COMPREHENSIVE STRESS TEST")
print("=" * 60)

# --- 1. Health & Diagnostics ---
print("\n[1/9] Health & Diagnostics Validation")
status, elapsed, payload, err = make_request("GET", "/api/health")
print_result("Health Check", status, elapsed, err, f"Loaded: {payload.get('model_loaded') if payload else False}")

status, elapsed, payload, err = make_request("GET", "/api/diagnostics")
if payload:
    details = f"Nodes: {payload.get('nodes')}, Params: {payload.get('parameters')}, Model: {payload.get('model')}"
else:
    details = ""
print_result("Diagnostics", status, elapsed, err, details)

# --- 2. Graph Integrity ---
print("\n[2/9] Graph Integrity Test")
status, elapsed, graph_data, err = make_request("GET", "/api/graph")
test_node_id = None
if status == 200 and graph_data:
    nodes_count = len(graph_data.get('nodes', []))
    edges_count = len(graph_data.get('edges', []))
    bbox = graph_data.get('bbox', {})
    if nodes_count > 0:
        test_node_id = graph_data['nodes'][0]['node_id']
    details = f"Nodes: {nodes_count}, Edges: {edges_count}, BBox: {bbox.get('min_lat'):.2f} to {bbox.get('max_lat'):.2f}"
    print_result("Graph Data Load", status, elapsed, err, details)
    if nodes_count != 2290:
        print(f"  [WARN] Expected 2290 nodes, got {nodes_count}")
else:
    print_result("Graph Data Load", status, elapsed, err)

# --- 3. ST-GCN Simulation Deep Test ---
print("\n[3/9] ST-GCN Simulation Across Time")
hours_to_test = [0, 9, 12, 18, 23]
for hr in hours_to_test:
    status, elapsed, payload, err = make_request("POST", "/simulate", {
        "latitude": 12.9255, "longitude": 77.6186, "hour_of_day": hr, "day_of_week": 1
    })
    if payload and "nodes" in payload:
        congested = len([n for n in payload["nodes"] if n.get("congested")])
        details = f"Hour: {hr:02d}:00, Congested Nodes: {congested}/{len(payload['nodes'])}"
        print_result(f"Simulate H={hr}", status, elapsed, err, details)
    else:
        print_result(f"Simulate H={hr}", status, elapsed, err)

# --- 4. Predictive Node Analysis ---
print("\n[4/9] Per-Node Prediction Validation")
if test_node_id:
    status, elapsed, payload, err = make_request("GET", f"/api/predict/{test_node_id}?hour_of_day=14")
    if payload and "predicted_volumes" in payload:
        vols = payload["predicted_volumes"]
        details = f"Node: {test_node_id}, 3-hr Prediction: {[round(v, 2) for v in vols]}"
        print_result("Node Prediction", status, elapsed, err, details)
    else:
        print_result("Node Prediction", status, elapsed, err)
else:
    print("  [SKIP] No node available for test")

# --- 5. Ripple Cascade Validation ---
print("\n[5/9] Ripple Cascade Simulation")
if test_node_id:
    status, elapsed, payload, err = make_request("POST", "/api/ripple", {"node_id": test_node_id})
    if payload:
        method = payload.get("calculation_method")
        delay = payload.get("total_delay_minutes", 0)
        edges = len(payload.get("affected_edges", []))
        details = f"Method: {method}, Delay: {delay:.2f}m, Affected Edges: {edges}"
        print_result("Ripple Cascade", status, elapsed, err, details)
        if method == "dijkstra-fallback":
            print("  [WARN] Used Dijkstra fallback instead of ST-GCN model")
    else:
        print_result("Ripple Cascade", status, elapsed, err)
else:
    print("  [SKIP] No node available for test")

# --- 6. Anomaly Detection (YOLO / RF) ---
print("\n[6/9] Real-time Anomaly Polling")
anomalies_found = False
for i in range(3):
    status, elapsed, payload, err = make_request("GET", "/api/anomalies")
    if payload:
        anom_count = len(payload.get("anomalies", []))
        frame = payload.get("frame_index")
        details = f"Frame: {frame}, Anomalies Detected: {anom_count}"
        print_result(f"Poll Anomalies (Try {i+1})", status, elapsed, err, details)
        if anom_count > 0:
            anomalies_found = True
    else:
        print_result(f"Poll Anomalies (Try {i+1})", status, elapsed, err)
    time.sleep(0.5)

# --- 7. Dispatch Lifecycle ---
print("\n[7/9] Dispatch & Resolution Lifecycle")
if test_node_id:
    status, elapsed, payload, err = make_request("POST", "/api/dispatch", {"node_id": test_node_id})
    if payload and payload.get("success"):
        details = f"Dispatched Unit: {payload.get('dispatch', {}).get('dispatch_id')}"
        print_result("Dispatch Unit", status, elapsed, err, details)
    else:
        print_result("Dispatch Unit", status, elapsed, err)
        
    status, elapsed, payload, err = make_request("POST", "/api/reset")
    print_result("State Reset", status, elapsed, err)
else:
    print("  [SKIP] No node available for test")

# --- 8. Edge Cases & Validation ---
print("\n[8/9] Boundary & Error Handling Validation")
# Invalid hour
status, elapsed, payload, err = make_request("POST", "/simulate", {
    "latitude": 12.9, "longitude": 77.6, "hour_of_day": 25, "day_of_week": 1
})
if status == 400:
    print_result("Simulate Invalid Hour (25)", 200, elapsed, None, "Correctly rejected (400)")
else:
    print_result("Simulate Invalid Hour (25)", status, elapsed, "Should be 400")

# Invalid Node Ripple
status, elapsed, payload, err = make_request("POST", "/api/ripple", {"node_id": "INVALID_NODE_999"})
if status == 404:
    print_result("Ripple Invalid Node", 200, elapsed, None, "Correctly rejected (404)")
else:
    print_result("Ripple Invalid Node", status, elapsed, "Should be 404")

# Invalid Node Predict
status, elapsed, payload, err = make_request("GET", "/api/predict/INVALID_NODE_999")
if status == 404:
    print_result("Predict Invalid Node", 200, elapsed, None, "Correctly rejected (404)")
else:
    print_result("Predict Invalid Node", status, elapsed, "Should be 404")

# --- 9. Concurrency Stress Test ---
print("\n[9/9] Concurrency Load Testing")
def worker_simulate(i):
    return make_request("POST", "/simulate", {
        "latitude": 12.9, "longitude": 77.6, "hour_of_day": 12, "day_of_week": 1
    })

def worker_ripple(i):
    if not test_node_id: return 0, 0, None, "No node"
    return make_request("POST", "/api/ripple", {"node_id": test_node_id})

NUM_THREADS = 10
print(f"  Launching {NUM_THREADS} concurrent requests...")

t0 = time.time()
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = []
    for i in range(NUM_THREADS // 2):
        futures.append(executor.submit(worker_simulate, i))
        futures.append(executor.submit(worker_ripple, i))
    
    results = [f.result() for f in futures]

total_elapsed = time.time() - t0
success_count = sum(1 for r in results if r[0] == 200)

if success_count == NUM_THREADS:
    print_result(f"Concurrency ({NUM_THREADS} reqs)", 200, total_elapsed, None, f"All {success_count} succeeded")
else:
    print_result(f"Concurrency ({NUM_THREADS} reqs)", 500, total_elapsed, f"Only {success_count}/{NUM_THREADS} succeeded")

print("\n" + "=" * 60)
print("STRESS TEST COMPLETE")
print("=" * 60)
