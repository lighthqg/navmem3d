#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--route',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads(a.route.read_text());g=json.loads(a.graph.read_text());nodes={x['node_id']:x for x in g['nodes']};seq=r['route']['node_sequence'];w,h=256,192;out=Image.new('RGB',(len(seq)*w, h+46),(15,19,25));d=ImageDraw.Draw(out)
for i,nid in enumerate(seq):
 n=nodes[nid];im=Image.open(n['reference_rgb_uri']).convert('RGB');im.thumbnail((w,h));x=i*w;out.paste(im,(x+(w-im.width)//2,(h-im.height)//2));d.rectangle((x,h,x+w,h+46),fill=(5,8,12));d.text((x+6,h+5),f'{i+1}. {nid}',fill='white');d.text((x+6,h+23),n['source_frame_id'],fill=(170,185,195))
 if i<len(seq)-1:d.text((x+w-13,h//2),'→',fill=(75,230,190),stroke_width=1)
a.output.parent.mkdir(parents=True,exist_ok=True);out.save(a.output)
