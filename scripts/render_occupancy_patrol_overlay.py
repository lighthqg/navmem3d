#!/usr/bin/env python3
"""Render a generated three-state occupancy map with its actual patrol trace."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

p=argparse.ArgumentParser();p.add_argument('--occupancy',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--patrol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
state=np.asarray(Image.open(a.occupancy).convert('L'));meta=json.loads(a.metadata.read_text());frames=json.loads(a.patrol.read_text())['frames']
canvas=np.empty((*state.shape,3),np.uint8);canvas[state==255]=(245,247,250);canvas[state==0]=(20,23,28);canvas[state==127]=(128,132,138)
origin=np.asarray(meta['origin_xy']);scale=float(meta['scale_m']);poses=np.asarray([f['camera_position'] for f in frames]);pix=np.floor((poses[:,:2]-origin)/scale).astype(int)
image=Image.fromarray(canvas).resize((state.shape[1]*2,state.shape[0]*2),Image.Resampling.NEAREST);draw=ImageDraw.Draw(image)
path=[(int(x*2),int(y*2)) for x,y in pix]
draw.line(path,fill=(255,80,110),width=2)
for i in range(0,len(path),25):
 x,y=path[i];draw.ellipse((x-3,y-3,x+3,y+3),fill=(255,188,40),outline=(20,20,20))
draw.text((10,10),'white free | black occupied | gray unknown | pink patrol',fill='white',stroke_width=2,stroke_fill='black')
a.output.parent.mkdir(parents=True,exist_ok=True);image.save(a.output)
