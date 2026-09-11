# 第一版交付：单 Marble 房间的观测拓扑语义导航

正式闭环产物见 [`observed_map/README.zh-CN.md`](observed_map/README.zh-CN.md)。第一版将空间观测层与已验证的巡视无向拓扑骨架并行保存，再通过统一接口叠加；后续可在深度质量达标后替换拓扑编译器。

空间层和拓扑层可以通过 `scripts/render_layered_map.py` 统一叠加，示例见 [`layered_space_topology.png`](layered_space_topology.png)。当前工程基线仍使用已验证的巡视拓扑骨架；观测 occupancy 先作为空间解释层，待深度质量满足验收后再替换或增强拓扑编译器。

## 系统问题与边界

目标是：机器人先巡视一个房间，巡视 RGB 送入 Marble 构建 World B；测试时从**协议给定的已知路线节点**出发，依据语言目标沿自己走过的视觉路线到达目标观察位。系统不读取仿真 occupancy、全局坐标、完整 A Gaussian、标签或结构真值。

第一版不包含未知随机起点、自主巡视、全局稠密地图、多 Marble 拼接或 B→真实世界的米制配准。

## 已交付的系统资产

| 文件 | 用途 |
|---|---|
| `visual_route_graph.json` | 22 个高纹理关键帧节点、21 条已实际巡视的可逆边；不含全局坐标与 occupancy。 |
| `visual_route_nodes.jpg` | 机器人可见的视觉路线节点总览。 |
| `entity_route_binding_local.json` | B 中实体到巡视节点的绑定。采用 B 代表虚拟视图与巡视 RGB 的 SIFT + RANSAC 局部几何证据；33/48 实体达到至少 6 个内点，可作为首版路线目标。 |
| `visual_route_replay_evaluation.json` | 非节点巡视帧的视觉重定位回放。 |
| `known_start_dining_table_replay.json` | 给定 `node_000`、查询“餐桌”的端到端路线计划和回放。 |
| `dining_table_visual_route.jpg` | 上述餐桌任务的 12 个参考路线节点。 |

## 运行时数据流

```text
语言目标 → Marble B Entity Index → 可导航实体绑定节点
给定 start_node → Visual Route Graph 图搜索 → 参考帧序列
当前 RGB → SIFT/RANSAC 路线重定位 → 进度与下一边
目标节点附近的当前 RGB → 目标再次确认
```

`entity_route_binding_local.json` 中只有 `navigation_eligible: true` 的实体能生成导航路线。到达绑定节点只表示到达候选观察位；必须用当前 RGB 对目标做最后确认，才能报告成功。

## 本轮回放验证

1. **路线重定位：** 42 张未用作节点的巡视帧中，39 张达到局部视觉证据阈值（92.9%）；这 39 张均关联到正确路线边端点（100%）。
2. **餐桌任务：** 从已知 `node_000` 到绑定 `node_011` 的 12 个节点路线中，全部节点在自身 RGB 回放中被精确定位（12/12）。

这验证的是静态场景、相同巡视视觉域中的离线 repeatability，不是实机控制成功率。真实执行仍需接入轮速/VIO、控制器和局部安全停车；这些模块可消费同一 `visual_route_graph.json`，无需进入仿真真值地图。

## 生成命令

```bash
cd /home/hqg/Documents/Projects/DroidUp_/navmem3d
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli build-visual-route \
  --patrol artifacts/interiorgs_0001_839920/patrol/patrol.json \
  --renders artifacts/interiorgs_0001_839920/patrol/renders \
  --keyframe-stride 15 \
  --output artifacts/interiorgs_0001_839920/first_version/visual_route_graph.json
```
