# 本地数据

## 第一阶段：InteriorGS

主仿真环境使用InteriorGS + NavGSim。InteriorGS数据集约43.8 GB，Hugging Face仓库带访问条款，需要用户登录并接受条款后才能下载。单场景目录应保持官方结构：

```text
data/interiorgs/<scene_id>/
├── 3dgs_compressed.ply
├── labels.json
├── occupancy.png
├── occupancy.json
└── structure.json
```

下载后使用：

```bash
PYTHONPATH=src python3 -m navmem3d.cli register-interiorgs \
  --scene data/interiorgs/<scene_id> \
  --output data/worlds/interiorgs_<scene_id>/manifest.json
```

`labels.json`、`structure.json`与完整occupancy属于环境A的隐藏真值，只供轨迹生成和评测；机器人内部环境B必须从有限视角的Marble重建结果重新生成实体索引。

## 历史Smoke Test

场景资产和采集结果不提交版本库。当前开发机已准备 Habitat 官方 `van-gogh-room.glb` 与对应 navmesh。

官方推荐下载方式：

```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

该测试集无语义标注，只用于验证 Habitat RGB-D 巡视采集、关键帧选择和世界构建输入。

`worlds/public_room/room-7k.splat` 是公开 room 3DGS 测试资产，用于验证 `.splat` 解析、虚拟视角和逐像素 Gaussian ID 关联。其来源与校验值记录在同目录 `manifest.json`；该资产不随代码提交。
