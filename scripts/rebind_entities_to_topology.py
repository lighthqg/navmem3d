#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--entities',type=Path,required=True);ap.add_argument('--topology',type=Path,required=True);ap.add_argument('--patrol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 e=json.loads(a.entities.read_text());t=json.loads(a.topology.read_text());p=json.loads(a.patrol.read_text())
 frame_idx={f['frame_id']:i for i,f in enumerate(p['frames'])}; topo=t['nodes']
 # Topology nodes carry the source patrol frame selected for execution.
 out=[]
 for ent in e['entities']:
  cands=ent.get('route_candidates_local',[])
  if cands:
   cands=sorted(cands,key=lambda x:(x.get('homography_inliers',0),x.get('ratio_matches',0)),reverse=True)
   frame_id=cands[0].get('source_frame_id')
  else: frame_id=ent.get('selected_source_frame_id')
  if frame_id not in frame_idx: continue
  fi=frame_idx[frame_id]
  # A frame index is only a temporal hint.  Use the nearest spatial topology
  # node when the topology carries source patrol pixels; this avoids mapping
  # an entity seen late in the video onto an unrelated frame with a similar
  # timestamp after route loops.
  src_frame=p['frames'][fi]
  if src_frame.get('pixel'):
   target=min(topo,key=lambda n:(int(n.get('pixel',[0,0])[0])-int(src_frame['pixel'][0]))**2+(int(n.get('pixel',[0,0])[1])-int(src_frame['pixel'][1]))**2)
  else:
   target=min(topo,key=lambda n:abs(int(n.get('reference_frame_index',0))-fi))
  rec={'entity_id':ent['entity_id'],'labels':ent.get('labels',[]),'category_id':ent.get('category_id'),'source_virtual_view_index':ent.get('source_virtual_view_index'),'source_frame_id':frame_id,'source_frame_index':fi,'topology_node_id':target['node_id'],'topology_reference_frame_id':target.get('reference_frame_id'),'topology_reference_frame_index':target.get('reference_frame_index'),'local_visual_inliers':ent.get('local_visual_inliers',0),'binding_confidence':ent.get('binding_confidence',0.0),'navigation_eligible':bool(ent.get('navigation_eligible',False))}
  out.append(rec)
 doc={'schema_version':'0.1','protocol':'marble_entity_to_topology_node_rebinding','source_entities':str(a.entities.resolve()),'source_topology':str(a.topology.resolve()),'selection':'highest local SIFT/RANSAC evidence, then nearest topology reference frame','entities':out}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'entities':len(out),'eligible':sum(x['navigation_eligible'] for x in out)},ensure_ascii=False))
if __name__=='__main__':main()
