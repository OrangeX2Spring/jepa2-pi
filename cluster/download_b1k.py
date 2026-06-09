import os, sys
sys.path.insert(0, '/tmp/hflib')
from huggingface_hub import snapshot_download

TASK_FOLDERS = ["task-0000", "task-0020", "task-0029", "task-0035", "task-0042"]
patterns = []
for task in TASK_FOLDERS:
    patterns += [
        f"data/{task}/*",
        f"videos/{task}/observation.images.rgb.head/*",
        f"videos/{task}/observation.images.rgb.left_wrist/*",
        f"videos/{task}/observation.images.rgb.right_wrist/*",
        f"videos/{task}/observation.images.seg_instance_id.head/*",
        f"videos/{task}/observation.images.seg_instance_id.left_wrist/*",
        f"videos/{task}/observation.images.seg_instance_id.right_wrist/*",
        f"annotations/{task}/*",
        f"meta/episodes/{task}/*",
    ]
patterns += ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]

snapshot_download(
    repo_id="behavior-1k/2025-challenge-demos",
    repo_type="dataset",
    revision="v2.1",
    allow_patterns=patterns,
    local_dir="/mnt/projects/at3dcv/world_model/b1k_data",
    token=os.environ["HF_TOKEN"],
    max_workers=2,
)

