# 第三方依赖

本目录在开发机上用于放置外部仓库，但第三方源码不随 NavMem3D 提交。需要复现实验时，请按 `configs/third_party_lock.json` 中的仓库和 commit 单独获取：

- `gsplat`：GPU 3D Gaussian Splatting 渲染；
- `OpenSplat3D`：实例/语言语义增强参考实现；
- `SAM2`：实例分割模型；
- `spz`：SuperSplat 压缩资产工具；
- `NavGSim`：历史导航仿真参考。

各项目遵循自己的许可证和使用条款。
