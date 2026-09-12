#!/usr/bin/env python3
"""Build a local, static dashboard for a source-only closed-loop run."""
from __future__ import annotations
import argparse, html, json, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--m-dir', type=Path, required=True)
p.add_argument('--b-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--route', type=Path)
a = p.parse_args()

def read(path): return json.loads(path.read_text())
def relative(path): return os.path.relpath(path, a.output).replace(os.sep, '/')

metadata = read(a.m_dir / 'occupancy_metadata.json')
topology = read(a.m_dir / 'patrol_topology.json')
route = read(a.route or a.m_dir / 'route_to_sofa.json')
report = read(a.m_dir / 'closed_loop_self_check.json')
entities = read(a.b_dir / 'entities_to_m.json')['entities']
target_id = route['target_entity']['entity_id']
for entity in entities:
    entity['crop_uri'] = relative(a.b_dir / 'crops' / 'crops' / f"{entity['entity_id']}.png")
route['map_uri'] = relative((a.route or a.m_dir / 'route_to_sofa.json').with_suffix('.png'))
data = {'metadata': metadata, 'topology': topology, 'route': route, 'report': report, 'entities': entities}
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'dashboard.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
(a.output / 'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NavMem3D 闭环结果</title><style>
*{box-sizing:border-box}body{margin:0;background:#101317;color:#edf2f7;font:15px system-ui,sans-serif}header,main{max-width:1450px;margin:auto;padding:24px}header{border-bottom:1px solid #303842}h1{margin:0 0 8px}.note{color:#b7c5d3;line-height:1.6}.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:18px}.panel{background:#1a2027;border:1px solid #303842;border-radius:12px;padding:16px;margin:18px 0}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{background:#242d37;padding:10px 13px;border-radius:9px}.ok{color:#73e6a1}.warn{color:#ffca65}img.map{width:100%;image-rendering:auto;border-radius:8px;background:#80848a}.entity{display:grid;grid-template-columns:92px 1fr;gap:12px;border-top:1px solid #303842;padding:10px 0}.entity img{width:92px;height:70px;object-fit:cover;border-radius:6px;background:#111}.target{border-left:4px solid #ff5876;padding-left:8px}.muted{color:#a8b4c0}code{color:#97d6ff}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style><body><header><h1>NavMem3D：Marble 语义 → 机器人 M 地图 → 路线</h1><div class="note">此页面是离线闭环审阅件。空间与路线只来自机器人巡视和 A 的 3DGS；Marble B 只提供视觉语义候选。页面没有读取原生 occupancy、navmesh、碰撞器或语义标签。</div></header><main id="app"></main><script>
fetch('dashboard.json').then(r=>r.json()).then(d=>{const q=d.route, s=d.report.checks, ent=d.entities;const target=q.target_entity.entity_id;document.querySelector('#app').innerHTML=`
<div class="stats"><div class="stat">M 占据：${s.occupancy_counts['255']} free / ${s.occupancy_counts['0']} occupied / ${s.occupancy_counts['127']} unknown</div><div class="stat">拓扑：${d.topology.nodes.length} 节点 / ${d.topology.edges.length} 无向边</div><div class="stat">语义：${ent.length} 个 B→M 候选绑定</div><div class="stat ${d.report.passed?'ok':'warn'}">自检：${d.report.passed?'通过':'失败'}</div></div>
<div class="grid"><section class="panel"><h2>机器人 M：三态空间层 + 巡视拓扑 + 主路线</h2><img class="map" src="${q.map_uri}"><p class="note">蓝线是已经观察到的无向巡视边；青线是 <code>${q.query_term}</code> 的主候选路线，${q.cost_m.toFixed(2)} m。黑色为从 Gaussian 表面密度得到的障碍轮廓，灰色为未观察区域。</p></section><section class="panel"><h2>语义交接</h2><p>目标：<code>${target}</code> / ${q.target_entity.labels.join('、')}</p><p>主节点：<code>${q.target_topology_node_id}</code>；视觉候选：${q.target_candidate_routes.map(x=>'<code>'+x.topology_node_id+'</code> ('+(x.visual_similarity??0).toFixed(3)+')').join(' → ')}</p><p class="note">执行到主候选后必须用当前 RGB 确认目标。确认失败才按候选顺序重定位；不会把 B 的坐标直接当作 A/M 的控制坐标。</p></section></div>
<section class="panel"><h2>Marble B 实体候选</h2><p class="note">以下是多视角分割、3D 融合后，用实体代表 crop 在巡视 RGB 序列检索得到的节点候选。高重复物体会保留多个假设。</p>${ent.map(e=>`<article class="entity ${e.entity_id===target?'target':''}"><img src="${e.crop_uri}" loading="lazy"><div><b>${e.entity_id}</b> · ${e.labels.join(' / ')}<br><span class="muted">${(e.candidate_topology_nodes||[]).map(c=>c.topology_node_id+' '+(c.similarity??0).toFixed(3)).join(' · ')}</span></div></article>`).join('')}</section>`})</script></main></body></html>''')
print(a.output / 'index.html')
