#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
from PIL import Image,ImageDraw

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--topology',type=Path,required=True);ap.add_argument('--entities',type=Path,required=True);ap.add_argument('--route',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();t=json.loads(a.topology.read_text());e=json.loads(a.entities.read_text());W=1200;H=900; xs=[n['pixel'][0] for n in t['nodes']];ys=[n['pixel'][1] for n in t['nodes']];xmin,xmax=min(xs)-10,max(xs)+10;ymin,ymax=min(ys)-10,max(ys)+10;s=min((W-80)/(xmax-xmin),(H-80)/(ymax-ymin));pp=lambda p:(40+(p[0]-xmin)*s,H-40-(p[1]-ymin)*s); im=Image.new('RGB',(W,H),'#6f7275');d=ImageDraw.Draw(im)
 for ed in t['edges']: d.line([pp(p) for p in ed['polyline_px']],fill='#70d8ef',width=2)
 route_nodes=set();
 if a.route and a.route.exists(): route_nodes=set(json.loads(a.route.read_text()).get('node_sequence',[]))
 for n in t['nodes']:
  x,y=pp(n['pixel']); col='#ffdf52' if n['node_id'] in route_nodes else '#ff8b2b';d.ellipse((x-5,y-5,x+5,y+5),fill=col);d.text((x+6,y-5),n['node_id'].replace('topo_',''),fill='white')
 node_counts={}
 for z in e['entities']:
  n=z.get('topology_node_id');node_counts[n]=node_counts.get(n,0)+1
 for n,c in node_counts.items():
  q=next((x for x in t['nodes'] if x['node_id']==n),None)
  if q:d.text((pp(q['pixel'])[0]+6,pp(q['pixel'])[1]+8),f'E{c}',fill='#ffb3b3')
 d.text((20,20),f'Observed topology / entities={len(e["entities"])} / route highlighted={len(route_nodes)}',fill='white');a.output.parent.mkdir(parents=True,exist_ok=True);im.save(a.output)
if __name__=='__main__':main()
