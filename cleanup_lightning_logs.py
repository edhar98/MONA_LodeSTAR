import yaml
import os
import shutil
from pathlib import Path

yaml_path = "trained_models_summary.yaml"
logs_path = "lightning_logs"

with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

model_ids = set()
for particle_type, config in data.items():
    checkpoint = config.get('checkpoint_path', '')
    if checkpoint:
        model_id = checkpoint.split('/')[1]
        model_ids.add(model_id)
    
    for additional in config.get('additional_models', []):
        checkpoint = additional.get('checkpoint_path', '')
        if checkpoint:
            model_id = checkpoint.split('/')[1]
            model_ids.add(model_id)

print(f"Found {len(model_ids)} model IDs in YAML:")
for mid in sorted(model_ids):
    print(f"  {mid}")

logs_dirs = [d for d in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, d))]
print(f"\nFound {len(logs_dirs)} directories in lightning_logs")

to_delete = [d for d in logs_dirs if d not in model_ids]
to_keep = [d for d in logs_dirs if d in model_ids]

print(f"\nKeeping {len(to_keep)} directories")
print(f"Deleting {len(to_delete)} directories:")
for d in sorted(to_delete):
    print(f"  {d}")

response = input(f"\nProceed with deletion of {len(to_delete)} directories? (yes/no): ")
if response.lower() == 'yes':
    for d in to_delete:
        dir_path = os.path.join(logs_path, d)
        shutil.rmtree(dir_path)
        print(f"Deleted: {d}")
    print(f"\nCleanup complete. Deleted {len(to_delete)} directories.")
else:
    print("Cancelled.")

