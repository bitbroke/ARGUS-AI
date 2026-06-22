# -*- coding: utf-8 -*-
"""
Stress Test Script - Tests every API endpoint that the frontend depends on.
"""
import urllib.request
import json
import time
import sys
import os

# Fix Windows console encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

BASE = "http://localhost:8000"
test_node_id = None

def test(method, path, body=None, label=""):
    url = f"{BASE}{path}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=30)
        elapsed = time.time() - t0
        payload = json.loads(resp.read().decode())
        print(f"  [PASS] [{method}] {path} - {resp.status} OK ({elapsed:.2f}s)")
        return payload
    except Exception as e:
        print(f"  [FAIL] [{method}] {path} - FAILED: {e}")
        return None

print("=" * 60)
print("NEXUS FLOW - FULL STRESS TEST")
print("=" * 60)

# 1. Health Check
print("\n[1/6] Health Check")
r = test("GET", "/api/health")
if r:
    print(f"       model_loaded={r.get('model_loaded')}")

# 2. Graph Endpoint
print("\n[2/6] Graph Topology")
r = test("GET", "/api/graph")
if r:
    print(f"       nodes={len(r.get('nodes',[]))}, edges={len(r.get('edges',[]))}, corridor={r.get('corridor_name')}")
    first_node = r['nodes'][0] if r.get('nodes') else None
    if first_node:
        test_node_id = first_node['node_id']
        print(f"       Using node '{test_node_id}' for subsequent tests")

# 3. Simulate Endpoint
print("\n[3/6] ST-GCN Simulation (/simulate)")
r = test("POST", "/simulate", {"latitude": 12.9255, "longitude": 77.6186, "hour_of_day": 9, "day_of_week": 1})
if r:
    nodes = r.get("nodes", [])
    congested = [n for n in nodes if n.get("congested")]
    print(f"       total_nodes={len(nodes)}, congested={len(congested)}")
    if congested:
        test_node_id = congested[0]['node_id']
        print(f"       First congested node: {test_node_id} (vol={congested[0]['predicted_volume']:.2f})")

# 4. Anomalies Endpoint
print("\n[4/6] Anomalies Polling")
r = test("GET", "/api/anomalies")
if r:
    print(f"       anomalies={len(r.get('anomalies',[]))}, frame={r.get('frame_index')}")

# 5. Ripple Endpoint (the one that was crashing!)
print("\n[5/6] Ripple Cascade Calculation (/api/ripple)")
if test_node_id:
    r = test("POST", "/api/ripple", {"node_id": test_node_id})
    if r:
        print(f"       method={r.get('calculation_method')}")
        print(f"       delay={r.get('total_delay_minutes',0):.1f} min")
        print(f"       affected_streets={len(r.get('affected_street_names',[]))}")
        print(f"       affected_edges={len(r.get('affected_edges',[]))}")
else:
    print("  [SKIP] No test node available")

# 6. Dispatch Endpoint
print("\n[6/6] Dispatch Unit (/api/dispatch)")
if test_node_id:
    r = test("POST", "/api/dispatch", {"node_id": test_node_id})
    if r:
        print(f"       success={r.get('success')}, message={r.get('message')}")
else:
    print("  [SKIP] No test node available")

# 7. Diagnostics
print("\n[BONUS] Diagnostics")
r = test("GET", "/api/diagnostics")
if r:
    print(f"       model={r.get('model')}, loss={r.get('loss')}")

print("\n" + "=" * 60)
print("STRESS TEST COMPLETE")
print("=" * 60)
