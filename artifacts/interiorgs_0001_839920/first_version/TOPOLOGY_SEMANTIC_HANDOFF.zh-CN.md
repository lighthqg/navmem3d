# Marble 语义与机器人观测拓扑的交接

## 系统边界

本方案有三个独立层次：

1. **评测场地 A（InteriorGS）**：只负责渲染机器人巡视时能看到的 RGB、深度/雷达观测和传感器位姿，并提供隐藏真值用于评测。
2. **Marble World B**：接收巡视 RGB（可附带深度），生成高质量三维语义世界，输出实体类别、属性、实例和虚拟观察视角。
3. **机器人地图 M**：只消费巡视时的 RGB-D/LiDAR 与 VIO/轮速，自己建立 `free / occupied / unknown` 占据图，再从已观测的 `free` 区域提取无向拓扑图。

A 的 occupancy、labels、structure 和完整世界坐标不能作为系统输入。它们只用于评测路线、碰撞和目标真值。

## 数据流

```text
A 渲染 RGB + 深度/LiDAR + VIO
        ├───────────────┐
        ↓               ↓
机器人建图 M       Marble B
free/occupied/     3D 语义实体
unknown            与虚拟视角
        ↓               ↓
无向 Topology Graph ← 语义交接
        ↓
起点接入 + 目标接入
        ↓
近似最短路
        ↓
真实执行：当前 RGB/LiDAR/VIO + 控制器
```

## 语义交接记录

Marble 不向控制器发送坐标或速度，只把实体交给机器人地图中的观察节点：

```json
{
  "entity_id": "entity_0021",
  "topology_node_id": "topo_017",
  "preferred_frame_id": "frame_0162",
  "semantic_confidence": 0.92,
  "visual_binding_inliers": 7,
  "arrival_confirmation": "current_rgb"
}
```

`topology_node_id` 是机器人地图 M 中的目标观察位置；`preferred_frame_id` 是执行层用于视觉重定位的巡视参考帧。

## 双层地图输出

第一版保留两份独立数据：

```text
空间层：free / occupied / unknown + 障碍物边界
拓扑层：节点 + 无向边 + 通行折线 + 参考关键帧
```

二者通过统一像素坐标/地图变换叠加显示，但不会由显示器重新推断连通关系。空间层回答“哪里被占据、哪里未知”；拓扑层回答“哪些路线已经确认连通”。可复用渲染器为 `scripts/render_layered_map.py`，其输入是任意三态 occupancy、任意符合字段契约的拓扑 JSON，以及可选路线 JSON。

## 拓扑图与路线

拓扑图来自机器人自己的观测占据图，而不是巡视轨迹折线，也不是 A 的隐藏 occupancy：

```text
free      已观测且可通行
occupied  已观测且有障碍物
unknown   未观测
```

只允许从 `free` 提取节点和无向边，`unknown` 不得当作可通行。节点包括端点、转弯点、分叉/汇合点和长边采样点；每个节点绑定一张附近巡视关键帧。边保存实际通行折线、长度和安全裕量。

起点由当前 RGB/VIO 定位到 M 的拓扑节点，目标由 Marble 实体索引交接到一个或多个候选节点。对每个候选目标运行 Dijkstra/A*，第一版边代价为通行折线长度，并按导航资格、视觉证据和路径代价排序。

路线输出：

```json
{
  "node_sequence": ["topo_000", "topo_012", "topo_017"],
  "edge_sequence": ["corridor_000", "corridor_012"],
  "cost_m": 4.6
}
```

执行层沿边上的关键帧做视觉跟踪，到目标观察节点后必须用当前 RGB 再确认实体。

## 第一版实现顺序（当前状态）

1. 为 A 的巡视位姿生成与 RGB 对齐的深度/雷达观测；
2. 用观测建立机器人自己的三态 occupancy；
3. 从该 occupancy 提取无向拓扑和节点关键帧；
4. 将 Marble 实体重新绑定到 `topo_###`；
5. 输出并评测近似最短路和执行关键帧序列。

本版已完成单房间离线闭环，正式入口位于 `observed_map/`：

```text
rgbd_observed/robot_patrol_rgbd_manifest.json
        ↓
observed_map/observed_occupancy.png
        ↓
observed_map/topology_graph_observed.json
        ↓
observed_map/entity_topology_binding_observed.json
        ↓
observed_map/route_dining_table.json
```

当前 RGB-D 清单已经生成，但 CPU Gaussian center z-buffer 只是临时深度近似，不能直接作为正式占据建图结果。第一次拓扑尝试暴露了两个错误：稀疏高斯中心造成大量 unknown/occupied 空洞；在自由空间上按邻近节点连边会产生虚假捷径。正式验收必须先通过深度覆盖率、自由空间覆盖率、孤立节点数和连通分量检查，再生成拓扑。A 的隐藏 occupancy、labels、structure 仍只用于评测，不能用于修补这些输入。
