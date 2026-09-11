# 第一版观测拓扑闭环

这里是第一版双层地图的可复现产物。空间层由机器人观测接口生成；路线层采用已验证的巡视无向拓扑骨架。当前深度由高斯中心 z-buffer 近似生成，作为空间层诊断输入，不能替代真实 RGB-D/LiDAR 质量。A 的 `occupancy.png`、`labels.json`、`structure.json` 不进入机器人地图推理。

## 产物

- `../rgbd_observed/robot_patrol_rgbd_manifest.json`：300 帧 RGB-D/位姿清单。当前深度生成器是 CPU Gaussian center z-buffer fallback，接口与真实 RGB-D/LiDAR 输入一致。
- `observed_occupancy.png`、`observed_occupancy.npy`：机器人观测三态地图，0=occupied、127=unknown、255=free。
- `../topology_graph_final.json`：第一版路线骨架，41 个节点、48 条无向边，节点带巡视关键帧。
- `topology_graph_observed.json`、`.png`：从当前近似深度自动提拓扑的诊断实验，保留用于误差分析，不作为路线骨架。
- `entity_topology_binding_observed.json`：Marble B 的 48 个实体交接到 M 拓扑节点。
- `route_dining_table.json`：从 `topo_000` 查询 dining table 的候选路线和近似最短路。
- `entity_route_handoff.png`：实体节点与路线叠加可视化。

## 复现

```bash
PYTHONPATH=src .envs/semantic/bin/python scripts/render_patrol_depth_cpu.py \
  --asset data/interiorgs/0001_839920/3dgs_compressed.ply \
  --patrol artifacts/interiorgs_0001_839920/patrol/patrol.json \
  --rgb-dir artifacts/interiorgs_0001_839920/patrol/renders \
  --output-dir artifacts/interiorgs_0001_839920/first_version/rgbd_observed --stride 2

PYTHONPATH=src .envs/semantic/bin/python scripts/build_observed_occupancy.py \
  --manifest artifacts/interiorgs_0001_839920/first_version/rgbd_observed/robot_patrol_rgbd_manifest.json \
  --output-dir artifacts/interiorgs_0001_839920/first_version/observed_map --sample 8

PYTHONPATH=src .envs/semantic/bin/python scripts/build_observed_topology.py \
  --occupancy artifacts/interiorgs_0001_839920/first_version/observed_map/observed_occupancy.png \
  --metadata artifacts/interiorgs_0001_839920/first_version/observed_map/occupancy_metadata.json \
  --manifest artifacts/interiorgs_0001_839920/first_version/rgbd_observed/robot_patrol_rgbd_manifest.json \
  --output artifacts/interiorgs_0001_839920/first_version/observed_map/topology_graph_observed.json
```
