#!/usr/bin/env python3
"""Bind Marble entities to topology compiled from observed occupancy."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--entities',type=Path,required=True);ap.add_argument('--topology',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 src=json.loads(a.entities.read_text()); topo=json.loads(a.topology.read_text()); man=json.loads(a.manifest.read_text()); frames={f['frame_id']:f for f in man['frames']}; nodes=topo['nodes']; out=[]
 for e in src['entities']:
  fid=e.get('source_frame_id') or e.get('selected_source_frame_id'); fr=frames.get(fid)
  if fr:
   p=fr['camera_position']; node=min(nodes,key=lambda n:(n['position_xy'][0]-p[0])**2+(n['position_xy'][1]-p[1])**2)
  else: node=None
  rec=dict(e);rec.update({'topology_node_id':node['node_id'] if node else None,'topology_reference_frame_id':node.get('reference_frame_id') if node else None,'binding_space':'observed_occupancy_topology','binding_confidence':float(e.get('binding_confidence',0.0))*(1.0 if node else 0.0),'navigation_eligible':bool(e.get('navigation_eligible',False) and node)})
  out.append(rec)
 doc={'schema_version':'0.2','protocol':'marble_entity_to_observed_occupancy_topology','source_entities':str(a.entities.resolve()),'source_topology':str(a.topology.resolve()),'source_manifest':str(a.manifest.resolve()),'hidden_occupancy_consumed':False,'entities':out}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'entities':len(out),'bound':sum(x['topology_node_id'] is not None for x in out),'output':str(a.output)},ensure_ascii=False))
if __name__=='__main__':main()
