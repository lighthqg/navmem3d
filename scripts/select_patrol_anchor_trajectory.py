from __future__ import annotations
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--patrol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ids',type=int,nargs='+',required=True);a=p.parse_args();d=json.loads(a.patrol.read_text());lookup={int(x['frame_id'].split('_')[-1]):x for x in d['frames']};frames=[lookup[i] for i in a.ids];a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'schema_version':'0.1','world_role':'A','purpose':'semantic-landmark observations for B-to-A registration','frames':frames},ensure_ascii=False,indent=2)+'\n')
