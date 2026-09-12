# 两个 Marble 场景的无真值输入回归

本回归检验同一条“B 提语义、M 管路线”的链路能否在两个现有 Marble 导出场景运行。它不是对最终导航成功率的宣称：人工巡视路径已给定，输出只说明离线数据流、约束检查和候选路线在两个场景都可复现。

| 场景 | 词表 / 查询 | M：free / occupied / unknown | 拓扑 | B 端去重实体 | 示例主路线 | 自检 |
|---|---|---:|---:|---:|---:|---|
| `0001_839920` 餐厅 | restaurant vocab / `dining_table` | 4,604 / 3,513 / 3,188 | 26 节点、26 实际巡视边 | 71 | 5.43 m，4 节点 | 通过 |
| `0007_840137` 休息室 | nightclub vocab / `sofa` | 12,985 / 8,771 / 14,124 | 39 节点、39 实际巡视边 | 46 | 19.62 m，9 节点 | 通过 |

两次运行的共同约束：

- 只读取源 Gaussian、人工巡视轨迹、巡视 RGB 和 Marble 的 `world_b.ply`；
- 不读取原生 occupancy、labels、structure、navmesh 或 collider；
- M 的每个巡视栅格都重新检查为 `free`；
- 每个 B 实体都必须交接到至少一个 M 拓扑候选；
- 路线结果必须带有“当前 RGB 确认”到达策略。

命令：

```bash
NAVMEM3D_VOCAB=configs/restaurant_vocab.zh_en.json \
NAVMEM3D_QUERY_TERM=dining_table \
scripts/run_offline_closed_loop.sh 0001_839920

NAVMEM3D_VOCAB=configs/nightclub_vocab.zh_en.json \
NAVMEM3D_QUERY_TERM=sofa \
scripts/run_offline_closed_loop.sh 0007_840137
```

每次运行会生成 `m_closed_loop/dashboard/index.html`。这个本地页面是审阅接口，不输入规划器。
