# M 地图与 Marble 语义交接：第一版实现

第一版不把 Marble 当作碰撞仿真器，也不把 InteriorGS 的隐藏 occupancy、navmesh、碰撞体或标签当作系统输入。

```text
人工带领机器人巡视
  + 3DGS Gaussian 几何
        ↓
M：Gaussian KDE 三态占据图
  free / occupied / unknown
        ↓
M：仅由巡视轨迹形成的无向拓扑图
  节点、巡视参考帧、实际穿行边

Marble B 虚拟视角
  ↓ Grounding DINO + SAM2
实例、类别、代表 crop、Gaussian 支持区域
  ↓ entity crop → 巡视 RGB 的 CLIP 检索
视觉候选拓扑节点
        ↓
关键词 / 关键词 + 拓扑关系
        ↓
Dijkstra 巡视路线
        ↓
机器人当前 RGB 确认目标后到达
```

## 约束

- `unknown` 永远不作为路线自由空间。
- 拓扑边只来自机器人实际巡视；不得根据二维自由区补一条看似更短的边。
- Marble 到 M 的当前绑定是视觉检索候选，不能写成已完成的米级坐标配准。
- 每条实体路线到达后都要通过当前 RGB 重新确认。 

## 餐厅场景当前产物

`artifacts/interiorgs_0007_840137/m_kde_v1/` 含有：

- `observed_occupancy.png`：不读取原生图的 KDE 三态图；
- `patrol_topology.json`：39 节点、39 条无向巡视边；
- `topology_self_check.json`：巡线图连通性检查；
- `route_to_sofa.json` / `.png`：关键词路线样例。

`artifacts/interiorgs_0007_840137/world_b/semantic_guided/` 含有 Marble 世界的虚拟视角、SAM mask、融合实体索引和视觉交接候选。
