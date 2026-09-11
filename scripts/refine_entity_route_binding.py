#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2
p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--views',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--min-inliers',type=int,default=6);a=p.parse_args();b=json.loads(a.binding.read_text());g=json.loads(a.graph.read_text());v=json.loads(a.views.read_text());root=a.views.parent
sift=cv2.SIFT_create(nfeatures=1800);matcher=cv2.BFMatcher(cv2.NORM_L2);cache={}
def features(path):
 if path not in cache:
  im=cv2.imread(str(path),0);cache[path]=sift.detectAndCompute(im,None)
 return cache[path]
for e in b['entities']:
 im=root/v['views'][e['source_virtual_view_index']]['rgb_uri'];kp,desc=features(im);scored=[]
 for node in g['nodes']:
  kr,dr=features(Path(node['reference_rgb_uri']));good=[]
  if desc is not None and dr is not None:
   good=[x[0] for x in matcher.knnMatch(desc,dr,k=2) if len(x)==2 and x[0].distance<.72*x[1].distance]
  inn=0
  if len(good)>=4:
   import numpy as np
   q=np.float32([kp[x.queryIdx].pt for x in good]).reshape(-1,1,2);r=np.float32([kr[x.trainIdx].pt for x in good]).reshape(-1,1,2);_,mask=cv2.findHomography(q,r,cv2.RANSAC,5.0);inn=int(mask.sum()) if mask is not None else 0
  scored.append({'node_id':node['node_id'],'source_frame_id':node['source_frame_id'],'ratio_matches':len(good),'homography_inliers':inn})
 scored.sort(key=lambda x:(x['homography_inliers'],x['ratio_matches']),reverse=True);e['route_candidates_local']=scored[:3];best=scored[0];e['selected_route_node_id']=best['node_id'];e['selected_source_frame_id']=best['source_frame_id'];e['local_visual_inliers']=best['homography_inliers'];e['navigation_eligible']=best['homography_inliers']>=a.min_inliers;e['status']='route_bound_with_local_visual_evidence' if e['navigation_eligible'] else 'queryable_but_route_binding_insufficient'
b['method']='B_representative_virtual_view_to_patrol_keyframe_local_SIFT_homography';b['minimum_local_inliers']=a.min_inliers;b['navigation_eligible_entity_count']=sum(x['navigation_eligible'] for x in b['entities']);b['status']='candidate_bindings_require_arrival_confirmation';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(b,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'entities':len(b['entities']),'eligible':b['navigation_eligible_entity_count']}))
