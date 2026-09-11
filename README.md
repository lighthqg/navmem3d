# NavMem3D

NavMem3D 是一个面向机器人导航的工程原型：把一次巡视得到的多视角观测交给高质量三维重建器（例如 Marble），再把重建出的实体语义交接给机器人自己的空间地图和路线拓扑。

项目关注的是可复用的系统接口，而不是绑定某个供应商或某个导航仿真器。第一版以单房间离线闭环为目标，验证“巡视、三维重建、实体查询、拓扑路线、视觉执行”的连接方式。

## 三个世界

```text
A：评测场地（InteriorGS 或其他仿真器）
  提供机器人可见的 RGB、深度/LiDAR、里程计或 VIO
  隐藏 occupancy、labels、structure 只供评测器使用

B：数字孪生（Marble / 本地 3DGS）
  从巡视图像重建高质量三维世界
  输出实体、属性、实例和虚拟视角

M：机器人自己的地图
  从 RGB-D/LiDAR 建立 free / occupied / unknown
  从已观测自由空间提取无向拓扑图和路线关键帧
```

B 不直接向控制器发送坐标。语义编译器把 Marble 实体绑定到 M 的拓扑观察节点；导航器根据当前起点、关键词/属性/关系查询结果和拓扑边，计算近似最短路线。到达目标节点后，执行层必须用当前观测再次确认实体。

## 当前实现

- 世界包和巡视序列 JSON 契约，以及环境检查命令；
- 均匀选帧和基于位姿覆盖的关键帧选择；
- `.splat`、标准 3DGS PLY、SuperSplat compressed PLY 的检查；
- 本地 3DGS 虚拟视角渲染和逐像素 Gaussian contributor ID；
- Grounding DINO + SAM2 的固定词表实例编译，多视角融合、去重和代表视图；
- 中英文关键词、属性和显式空间关系查询；
- `free / occupied / unknown` 三态空间层与无向巡视拓扑层的联合渲染；
- 实体到拓扑观察节点的绑定，以及路线查询/评测接口。

正式方案和数据流见：

- [`docs/first_version_route_contract.md`](docs/first_version_route_contract.md)
- [`artifacts/interiorgs_0001_839920/first_version/TOPOLOGY_SEMANTIC_HANDOFF.zh-CN.md`](artifacts/interiorgs_0001_839920/first_version/TOPOLOGY_SEMANTIC_HANDOFF.zh-CN.md)
- [`artifacts/interiorgs_0001_839920/first_version/LAYERED_MAP_CONTRACT.zh-CN.md`](artifacts/interiorgs_0001_839920/first_version/LAYERED_MAP_CONTRACT.zh-CN.md)

## 快速开始

仓库内的示例是小型契约 fixture，不包含真实场景或模型：

```bash
python3 -m pip install -e .
PYTHONPATH=src python3 -m navmem3d.cli inspect-world examples/world_minimal/manifest.json
PYTHONPATH=src python3 -m navmem3d.cli select-frames \
  examples/capture_sequence.json --count 3 --strategy pose-coverage \
  --output /tmp/selected.json
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

联合地图渲染器接受任意符合契约的三态 occupancy、拓扑图和可选路线：

```bash
PYTHONPATH=src python3 scripts/render_layered_map.py \
  --occupancy <observed_occupancy.png> \
  --topology <topology_graph.json> \
  --route <route.json> \
  --output <layered_map.png>
```

需要 CUDA、3DGS 或语义模型时，使用项目环境和 `scripts/run_gpu.sh`。这些依赖不会随仓库提交；版本锁定信息见 [`configs/third_party_lock.json`](configs/third_party_lock.json)。

## 数据与复现边界

真实场景、Marble 导出资产、模型权重、视频、深度序列、渲染缓存和第三方源码都不进入 Git。下载方式、目录契约和 `evaluator_only` 边界见 [`data/README.md`](data/README.md)。仓库中的示例 JSON/PLY 只用于检查接口，不代表导航效果。

当前第一版是单房间离线原型。在线实时重建、跨房间 Marble 拼接、自动全局重定位和真实机器人执行仍是后续工作；任何使用仿真隐藏真值的脚本都必须明确标记为评测或接线测试。

## 许可证

代码采用 Apache-2.0。第三方模型、数据集和依赖遵循各自许可证。
