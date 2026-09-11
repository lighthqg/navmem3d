#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();g=json.loads(a.graph.read_text());nodes=g['nodes'];w,h=256,192;cols=4;rows=(len(nodes)+cols-1)//cols
out=Image.new('RGB',(cols*w,rows*(h+28)),(16,20,26));d=ImageDraw.Draw(out)
for i,n in enumerate(nodes):
 im=Image.open(n['reference_rgb_uri']).convert('RGB');im.thumbnail((w,h));x=(i%cols)*w;y=(i//cols)*(h+28);out.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2));d.rectangle((x,y+h,x+w,y+h+28),fill=(8,10,14));d.text((x+6,y+h+6),f"{n['node_id']} · {n['source_frame_id']}",fill='white')
a.output.parent.mkdir(parents=True,exist_ok=True);out.save(a.output)
