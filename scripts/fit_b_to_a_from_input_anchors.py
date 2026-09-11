from __future__ import annotations
import argparse,json,random
from pathlib import Path
import cv2,numpy as np
from plyfile import PlyData

def umeyama(s,t):
 ms,mt=s.mean(0),t.mean(0);a=s-ms;b=t-mt;u,sv,vt=np.linalg.svd(b.T@a/len(s));d=np.eye(3)
 if np.linalg.det(u@vt)<0:d[-1,-1]=-1
 r=u@d@vt;scale=(sv*np.diag(d)).sum()/max((a*a).sum()/len(s),1e-12);tr=mt-scale*r@ms;m=np.eye(4);m[:3,:3]=scale*r;m[:3,3]=tr;return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--anchors',type=Path,required=True);p.add_argument('--b-renders',type=Path,required=True);p.add_argument('--b-asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stride',type=int,default=5);p.add_argument('--min-inliers',type=int,default=12);p.add_argument('--threshold',type=float,default=.5);a=p.parse_args();doc=json.loads(a.anchors.read_text());frames=doc['frames'][::a.stride];v=PlyData.read(str(a.b_asset)).elements[0];xyz=np.stack([v[n] for n in ('x','y','z')],1).astype(np.float32);sift=cv2.SIFT_create(nfeatures=1800);matcher=cv2.BFMatcher();good=[];attempts=[]
 for f in frames:
  i=f['marble_video_frame'];bp=a.b_renders/f'frame_{i:04d}.png';ip=Path(f['rgb_uri']);ids=a.b_renders/f'frame_{i:04d}_top_ids.npy';rec={'frame':i,'source_patrol_frame_id':f['source_patrol_frame_id']}
  bi=cv2.imread(str(bp),0);ai=cv2.imread(str(ip),0);kb,db=sift.detectAndCompute(bi,None);ka,da=sift.detectAndCompute(ai,None)
  if db is None or da is None:attempts.append(rec);continue
  ms=[x[0] for x in matcher.knnMatch(db,da,k=2) if len(x)==2 and x[0].distance<.72*x[1].distance];rec['ratio_matches']=len(ms);top=np.load(ids)[...,0];p3=[];p2=[];bad=np.iinfo(np.uint32).max
  for q in ms:
   x,y=map(round,kb[q.queryIdx].pt)
   if 0<=x<top.shape[1] and 0<=y<top.shape[0]:
    gid=int(top[y,x])
    if gid!=bad and gid<len(xyz):p3.append(xyz[gid]);p2.append(ka[q.trainIdx].pt)
  rec['correspondences']=len(p3)
  if len(p3)>=a.min_inliers:
   K=np.array([[320,0,320],[0,320,240],[0,0,1]],float);ok,rv,tv,ins=cv2.solvePnPRansac(np.asarray(p3),np.asarray(p2),K,None,iterationsCount=400,reprojectionError=4,confidence=.999,flags=cv2.SOLVEPNP_EPNP);n=0 if ins is None else len(ins);rec['pnp_inliers']=n;rec['pnp_success']=bool(ok)
   if ok and n>=a.min_inliers: R=cv2.Rodrigues(rv)[0];rec['camera_position_b']=(-R.T@tv.reshape(3)).tolist();rec['camera_position_a']=f['camera_pose_world_a']['camera_position'];good.append(rec)
  attempts.append(rec)
 if len(good)<3: raise RuntimeError(f'only {len(good)} usable anchors')
 s=np.array([x['camera_position_b'] for x in good]);t=np.array([x['camera_position_a'] for x in good]);rng=random.Random(0);best=[];bm=None
 for _ in range(2000):
  ix=rng.sample(range(len(good)),3);m=umeyama(s[ix],t[ix]);e=np.linalg.norm((m[:3,:3]@s.T).T+m[:3,3]-t,axis=1);ii=np.where(e<a.threshold)[0]
  if len(ii)>len(best):best=ii;bm=m
 m=umeyama(s[best],t[best]);e=np.linalg.norm((m[:3,:3]@s.T).T+m[:3,3]-t,axis=1)
 for x,z in zip(good,e):x['camera_center_residual_m']=float(z);x['sim3_inlier']=bool(z<a.threshold)
 out={'schema_version':'0.1','method':'actual_input_frame_gaussian_pnp_ransac_umeyama_sim3','status':'candidate_requires_held_out_validation','source_input_anchors':str(a.anchors.resolve()),'source_b_asset':str(a.b_asset.resolve()),'checked_frames':len(frames),'pnp_anchor_count':len(good),'sim3_inlier_count':int((e<a.threshold).sum()),'ransac_threshold_m':a.threshold,'median_residual_m':float(np.median(e)),'transform_b_to_a':m.reshape(-1).tolist(),'anchors':good,'top_attempts':sorted(attempts,key=lambda x:x.get('pnp_inliers',0),reverse=True)[:30]};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
