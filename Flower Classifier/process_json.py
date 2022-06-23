import json

#Load json function
def load_json(json_file):
    
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name