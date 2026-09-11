from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from navmem3d.world.supersplat import load_supersplat_compressed_ply

def voxel(points, size, limit=7000):
 keys=np.floor(points/size).astype(np.int32); _,ix=np.unique(keys,axis=0,return_index=True); out=points[ix]
 return out[::max(1,len(out)//limit+1)]
def nearest(src,tgt,chunk=256):
 idx=[]; ds=[]
 for i in range(0,len(src),chunk):
  d=((src[i:i+chunk,None,:]-tgt[None,:,:])**2).sum(2); j=d.argmin(1);idx.append(j);ds.append(np.sqrt(d[np.arange(len(j)),j]))
 return np.concatenate(idx),np.concatenate(ds)
def sim(s,t):
 ms,mt=s.mean(0),t.mean(0);a=s-ms;b=t-mt;u,sv,vt=np.linalg.svd(b.T@a/len(s));q=np.eye(3)
 if np.linalg.det(u@vt)<0:q[-1,-1]=-1
 r=u@q@vt;scale=(sv*np.diag(q)).sum()/max((a*a).sum()/len(s),1e-12);m=np.eye(4);m[:3,:3]=scale*r;m[:3,3]=mt-scale*r@ms;return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--a',type=Path,required=True);p.add_argument('--b',type=Path,required=True);p.add_argument('--init',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iters',type=int,default=30);a=p.parse_args()
 A=load_supersplat_compressed_ply(a.a).means.astype(float);v=PlyData.read(str(a.b)).elements[0];B=np.stack([v[x] for x in ('x','y','z')],1).astype(float);A=A[np.isfinite(A).all(1)];B=B[np.isfinite(B).all(1)]
 # Cap to robust spatial surface samples; tiny unbounded splats are suppressed by q01/q99.
 A=A[((A>=np.quantile(A,.01,axis=0))&(A<=np.quantile(A,.99,axis=0))).all(1)];B=B[((B>=np.quantile(B,.01,axis=0))&(B<=np.quantile(B,.99,axis=0))).all(1)]
 A=voxel(A,.10);B=voxel(B,.07)
 m=np.asarray(json.loads(a.init.read_text())['transform_b_to_a'],float).reshape(4,4);hist=[]
 for it in range(a.iters):
  X=(m[:3,:3]@B.T).T+m[:3,3]; ix,d=nearest(X,A); keep=d<=np.quantile(d,.65); corr=sim(X[keep],A[ix[keep]]);m=corr@m
  hist.append({'iteration':it,'median_nn_m':float(np.median(d)),'trimmed_mean_nn_m':float(d[keep].mean()),'kept':int(keep.sum()),'correction_scale':float(np.cbrt(abs(np.linalg.det(corr[:3,:3]))))})
 out={'schema_version':'0.1','method':'trimmed_point_to_point_icp_on_gaussian_means','status':'experimental_requires_render_validation','source_a':str(a.a.resolve()),'source_b':str(a.b.resolve()),'initialization':str(a.init.resolve()),'a_points':len(A),'b_points':len(B),'transform_b_to_a':m.reshape(-1).tolist(),'history':hist}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'a':len(A),'b':len(B),'first':hist[0],'last':hist[-1]},indent=2))
if __name__=='__main__':main()
