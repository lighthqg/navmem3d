# 0007_840137：单房间离线闭环评测记录

本记录描述当前可复现的第一版闭环。目的不是证明 Marble 可以直接控制机器人，而是验证一条严格分工的工程链路：**高质量三维重建负责实体语义，机器人自身地图负责可通行空间和路线。**

## 输入边界

| 世界 | 使用的输入 | 明确不使用的输入 |
|---|---|---|
| A / InteriorGS | `3dgs_compressed.ply`、人工带领巡视得到的 `patrol.json` 与巡视 RGB | 原生 `occupancy.png` / `occupancy.json`、labels、structure、navmesh、collider |
| B / Marble | 导出的 `world_b.ply` | B 到 A 的全局米制坐标变换 |
| M / 机器人地图 | A 的源 Gaussian、巡视轨迹和巡视 RGB | A 的任意隐藏真值 |

人工巡视是第一版允许的采集方式。它等价于真实部署中由操作者带机器人走一圈；本评测不把仿真器的自动巡线当作系统输入。

## 当前闭环

```text
A 的源 Gaussian + 人工巡视
  ├─ Gaussian 低位表面/垂直表面密度 → M: free / occupied / unknown
  └─ 巡视位姿 → M: 39 节点、39 条无向巡视边

Marble B 的 world_b.ply
  ├─ 8 个虚拟视角
  ├─ Grounding DINO + SAM2 固定词表分割
  └─ 多视角 3D 融合 → 46 个去重实体

实体代表 crop × 巡视 RGB 检索
  → 每个实体保留 3 个按视觉相似度排序的 M 拓扑候选
  → 关键词查询 → 主候选路线 + 备用路线
  → 到达后以当前 RGB 确认；失败才切换候选或重定位
```

## 本次结果

执行：

```bash
scripts/run_offline_closed_loop.sh 0007_840137
```

自检输出 `m_closed_loop/closed_loop_self_check.json`，当前结果：

- M 占据栅格：12,985 free、8,771 occupied、14,124 unknown；
- 巡视拓扑：39 个节点、39 条实际巡视无向边；所有 1,089 个巡视栅格单元均落在 M 的 free；
- B 端：8 个虚拟视角、139 个实例 mask、73 个融合实体、46 个去重实体；
- 46 个实体都有至少一个 B→M 视觉候选；
- `sofa` 示例从 `topo_000` 到主候选 `topo_008`，路线 9 节点、19.62 m，并保留 3 个按视觉相似度排序的候选节点；
- 自检通过，且 `hidden_occupancy_consumed`、`simulator_navmesh_consumed`、`simulator_collider_consumed`、`semantic_labels_consumed` 均为 `false`。

运行后可直接在本地打开：

```text
artifacts/interiorgs_0007_840137/m_closed_loop/dashboard/index.html
```

页面同时显示三态 M 地图、实际巡视拓扑、路线、实体 crop 和每个实体的候选节点。该页面用于审阅，不参与任何规划。

## 关键约束与尚未解决的问题

1. B→M 是图像检索交接，**不是**已解决的米制配准。B 的对象坐标不能直接用于控制。
2. 当前拓扑来自一次闭环巡视，因此路线在 M 已观测的 free 栅格中做 A*，因此可穿过巡视覆盖到但未作为关键节点保存的自由空间；unknown 和 occupied 始终不可通行。
3. 占据轮廓来自 Gaussian 密度和巡视通行证据，够用于安全边界和可视化审阅，尚不能代替真实 LiDAR/深度 SLAM 的实时建图。
4. 高重复物体会出现语义混淆。系统保留 top-3 候选，执行层以当前 RGB 确认，不能仅凭离线索引宣布到达。
5. 该场景是单房间原型；多房间、跨 Marble 世界拼接、未知起点重定位和真实机器人闭环是后续评测项。
