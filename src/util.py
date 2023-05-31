import json

def write_to_config(data, line_key):
    config_file: str = "./src/config.json"
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    config_data[line_key] = data

    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=4)
    
    print(f"Wrote {data} to {line_key} ")