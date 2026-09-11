#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--crops',type=Path,required=True);p.add_argument('--views',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
 import open_clip,torch
 from PIL import Image
 e=json.loads(a.entities.read_text());c=json.loads(a.crops.read_text());v=json.loads(a.views.read_text());g=json.loads(a.graph.read_text())
 cr={x['entity_id']:x for x in c['entities']}; nodes=g['nodes']; viewroot=a.views.parent
 model,_,prep=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',device=a.device);model.eval()
 def enc(paths):
  chunks=[]
  with torch.inference_mode():
   for k in range(0,len(paths),32):
    x=torch.stack([prep(Image.open(q).convert('RGB')) for q in paths[k:k+32]]).to(a.device);z=model.encode_image(x);chunks.append((z/z.norm(dim=-1,keepdim=True)).cpu())
  return torch.cat(chunks)
 node_paths=[Path(x['reference_rgb_uri']) for x in nodes];nodef=enc(node_paths)
 view_paths=[viewroot/x['rgb_uri'] for x in v['views']];viewf=enc(view_paths)
 sims=viewf@nodef.T;byview=[]
 for i in range(len(v['views'])):
  vals,idx=sims[i].topk(min(3,len(nodes)));byview.append([{'node_id':nodes[int(j)]['node_id'],'similarity':float(q),'source_frame_id':nodes[int(j)]['source_frame_id']} for q,j in zip(vals,idx)])
 bindings=[]
 for entity in e['entities']:
  rec=cr.get(entity['entity_id']);vi=rec.get('source_view_index') if rec else None
  if vi is None: continue
  candidates=byview[vi]
  bindings.append({'entity_id':entity['entity_id'],'labels':entity.get('labels',[]),'category_id':entity.get('category_id'),'source_virtual_view_index':vi,'route_candidates':candidates,'selected_route_node_id':candidates[0]['node_id'],'selected_source_frame_id':candidates[0]['source_frame_id'],'binding_confidence':candidates[0]['similarity'],'status':'candidate_requires_target_arrival_visual_confirmation'})
 out={'schema_version':'0.1','method':'representative_B_virtual_view_to_patrol_keyframe_clip_retrieval','world_role':'B_route_support','system_visible':True,'source_entity_index':str(a.entities.resolve()),'source_route_graph':str(a.graph.resolve()),'threshold_note':'similarity is a retrieval score, not geometric pose proof; target arrival must confirm visually','entities':bindings}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'entities':len(bindings),'median_confidence':float(torch.tensor([x['binding_confidence'] for x in bindings]).median())}))
if __name__=='__main__':main()
