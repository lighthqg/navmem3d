# 双层地图接口契约

## 目的

把空间观测和路线连通性作为两个独立层保存，并提供一个确定性的联合渲染入口。该接口适用于 InteriorGS、真实 RGB-D/LiDAR 记录以及后续其他仿真环境。

## 空间层 `observed_space_map`

输入为灰度 occupancy：

- `0`：已观测障碍物；
- `255`：已观测可通行；
- `127`：未知，不得被路线规划当作可通行。

空间层可以额外提供 `origin_xy`、`scale_m` 和传感器覆盖率。障碍物轮廓由 occupied 与其他状态的边界得到，显示只需近似轮廓，不要求 mesh。

## 拓扑层 `patrol_topology_graph`

拓扑 JSON 至少包含：

- `nodes[].node_id`、`nodes[].pixel`、`nodes[].reference_frame_id`；
- `edges[].from_node`、`edges[].to_node`、`edges[].undirected=true`；
- `edges[].polyline_pixel` 或端点；
- `edges[].cost` 或 `length_m`。

渲染器只绘制 JSON 中的边，不根据节点距离补边，因此不会把相邻节点误连成捷径。

## 联合渲染

```bash
PYTHONPATH=src .envs/semantic/bin/python scripts/render_layered_map.py \
  --occupancy <observed_occupancy.png> \
  --topology <topology_graph.json> \
  --route <route.json> \
  --output <layered_map.png>
```

输出同时包含空间底图、障碍物边界、拓扑节点、无向边和可选路线高亮；旁边生成同名 JSON，记录每一层的来源，便于复现实验。
