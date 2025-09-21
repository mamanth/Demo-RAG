import json, os
from datetime import datetime
from pathlib import Path

class JSONLogger:
    def __init__(self, out_file='logs/requests.jsonl'):
        p = Path(out_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.out_file = p

    def log(self, obj):
        obj = dict(obj)
        obj.setdefault('timestamp', datetime.utcnow().isoformat()+'Z')
        with open(self.out_file, 'a', encoding='utf8') as f:
            f.write(json.dumps(obj)+'\n')
