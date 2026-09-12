#!/usr/bin/env python3
"""Bind Marble entity crops to patrol topology by CLIP retrieval, without metric B→A alignment."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--crops',type=Path,required=True);p.add_argument('--patrol',type=Path,required=True);p.add_argument('--patrol-renders',type=Path,required=True);p.add_argument('--topology',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=2);p.add_argument('--top-k',type=int,default=3);p.add_argument('--device',default='cuda');a=p.parse_args()
import open_clip,torch
from PIL import Image
entities=json.loads(a.entities.read_text());crops=json.loads(a.crops.read_text());crop_root=a.crops.parent;patrol=json.loads(a.patrol.read_text())['frames'];topology=json.loads(a.topology.read_text())
by_entity={x['entity_id']:x for x in crops['entities']};frames=[x for x in patrol[::a.stride] if (a.patrol_renders/f"{x['frame_id']}.png").exists()];nodes=topology['nodes']
model,_,pre=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',device=a.device);model.eval()
def enc(paths):
 chunks=[]
 with torch.inference_mode():
  for i in range(0,len(paths),32):
   b=torch.stack([pre(Image.open(x).convert('RGB')) for x in paths[i:i+32]]).to(a.device);z=model.encode_image(b);chunks.append((z/z.norm(dim=-1,keepdim=True)).cpu())
 return torch.cat(chunks).numpy()
frame_features=enc([a.patrol_renders/f"{x['frame_id']}.png" for x in frames])
records=[]
for e in entities['entities']:
 crop=by_entity.get(e['entity_id'])
 if not crop:continue
 feature=enc([crop_root/crop['crop_uri']])[0]; score=frame_features@feature
 # retain distinct nearby patrol frames only once; prevents a single visual
 # moment sampled every 2 frames from consuming all candidates.
 selected=[]
 for j in np.argsort(-score):
  frame=frames[int(j)]; pos=np.asarray(frame['camera_position'][:2]);
  if any(np.linalg.norm(pos-np.asarray(x['position_xy']))<1.0 for x in selected):continue
  node=min(nodes,key=lambda n:(n['position_xy'][0]-pos[0])**2+(n['position_xy'][1]-pos[1])**2)
  selected.append({'source_frame_id':frame['frame_id'],'position_xy':pos.tolist(),'topology_node_id':node['node_id'],'similarity':float(score[j])})
  if len(selected)>=a.top_k:break
 if not selected:continue
 best=selected[0];records.append({**e,'source_frame_id':best['source_frame_id'],'topology_node_id':best['topology_node_id'],'binding_type':'masked_entity_crop_to_patrol_rgb_clip_retrieval','binding_confidence':best['similarity'],'candidate_topology_nodes':selected,'navigation_eligible':True,'execution_requirement':'current-RGB target confirmation required'})
out={'schema_version':'1.0','protocol':'marble_entity_crop_to_patrol_topology_retrieval','metric_b_to_a_registration_used':False,'hidden_occupancy_consumed':False,'source_entities':str(a.entities.resolve()),'source_crops':str(a.crops.resolve()),'source_patrol':str(a.patrol.resolve()),'source_topology':str(a.topology.resolve()),'model':'ViT-B-32/openai','stride':a.stride,'entities':records,'statistics':{'entities':len(records),'top_k':a.top_k}}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out['statistics'],ensure_ascii=False))
