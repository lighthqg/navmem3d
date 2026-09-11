#!/usr/bin/env python3
"""Overlay World-B fused entity IDs on the virtual gsplat views."""
from __future__ import annotations
import argparse, colorsys, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def color_for(text: str) -> tuple[int,int,int]:
    h = (sum((i + 1) * ord(c) for i,c in enumerate(text)) % 360) / 360
    return tuple(round(v * 255) for v in colorsys.hsv_to_rgb(h, .72, 1.0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--entity-index',required=True,type=Path)
    ap.add_argument('--views-dir',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    args=ap.parse_args()
    doc=json.loads(args.entity_index.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    font=ImageFont.load_default()
    by_view: dict[int,list[tuple[dict,dict]]] = {}
    cache={}
    for e in doc['entities']:
        for o in e['observations']:
            p=Path(o['proposal_index'])
            proposal_doc=cache.setdefault(p,json.loads(p.read_text()))
            proposal=next(x for x in proposal_doc['proposals'] if x['proposal_id']==o['proposal_id'])
            by_view.setdefault(o['view_index'],[]).append((e,proposal))
    records=[]
    for vi in range(8):
        source=args.views_dir/f'view_{vi:03d}.png'
        im=Image.open(source).convert('RGB').convert('RGBA')
        overlay=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(overlay)
        labels=[]
        for e,p in by_view.get(vi,[]):
            c=color_for(e['entity_id']); mask_path=Path(p['source_mask_uri'])
            mask=Image.open(mask_path).convert('L')
            # Dilated mask edge, without external CV dependency.
            edge=mask.filter(Image.Filter.MaxFilter(5)) if hasattr(Image,'Filter') else mask
            # Pillow ImageFilter is imported lazily below for compatibility.
            x,y,w,h=map(int,p['bbox_xywh'])
            d.rectangle((x,y,x+w,y+h),outline=(*c,220),width=2)
            labels.append((x,y,w,h,e))
        # labels after all boxes: dark backing avoids illegible text.
        for x,y,w,h,e in labels:
            label=f"{e['entity_id']} · {e['category_id']}"
            box=d.textbbox((x,y),label,font=font)
            ty=max(0,y-13); tw=box[2]-box[0]+5
            d.rectangle((x,ty,x+tw,ty+12),fill=(0,0,0,210))
            d.text((x+2,ty+1),label,font=font,fill=(*color_for(e['entity_id']),255))
        im=Image.alpha_composite(im,overlay).convert('RGB')
        out=args.output_dir/f'view_{vi:03d}_entities.png'; im.save(out)
        records.append({'view_index':vi,'image_uri':out.name,'entity_observations':len(labels)})
    (args.output_dir/'manifest.json').write_text(json.dumps({'source_entity_index':str(args.entity_index.resolve()),'views':records},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()
