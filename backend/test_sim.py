import urllib.request
import json

req = urllib.request.Request(
    'http://localhost:8000/simulate', 
    data=b'{"latitude": 12.925557, "longitude": 77.618665, "hour_of_day": 9, "day_of_week": 1}', 
    headers={'Content-Type': 'application/json'}
)
response = urllib.request.urlopen(req)
print('Response snippet:', response.read().decode('utf-8')[:500])
