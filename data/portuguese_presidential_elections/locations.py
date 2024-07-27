import json

# Create a Set of locations
locations = set()

json_data = json.load(open('./locations.json', 'r', encoding='utf-8'))

for location in json_data:
    
    if 'brasil' in location.lower():
        continue
    
    if 'portugal' not in location.lower():
        locations.add(location)

json.dump(list(locations), open('./locations-parsed.json', 'w',encoding='utf-8'), indent=4, ensure_ascii=False)