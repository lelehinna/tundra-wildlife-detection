# wildcam：苔原延时相机中的稀有动物检测（包括远处的小目标兽群）

**场景：** 固定安装的延时相机俯瞰开阔苔原，按固定间隔拍照，数据量可达数百万张。

**目标：** 找出极少数有动物的图片，例如 270 万张中只有 0.05%。难点在于远处的兽群：只有几个像素，对比度很低，紧贴着地平线。

![示例](docs/example.jpg)

## 方案

```
同一相机的图片按时间排序
 → 时段背景模型：同一时段（默认每 3 小时一段）最近 12 帧的逐像素中值
 → 光照校正：帧与背景的比值经中值滤波，抵消曝光变化和漂移的云影，保留小目标
 → 稳健 z 分数：|帧 − 背景| ÷ 该像素历史波动（MAD）
 → 地平线条带：由长期中值图估计地平线
     条带内：允许 2 px 的小斑点，相邻斑点合并成"兽群"
     地面：只保留较大的变化（近处动物）；天空：忽略（鸟、云）
 [+ 可选 MegaDetector 分块推理，负责近处清晰可见的动物]
 → 候选确认：冻结 ResNet 提取"当前帧裁剪图 + 背景裁剪图"的特征，加形状统计，用逻辑回归判断
 → 时间支持：相邻帧在合理位移内也出现候选，就加分（10 分钟间隔下目标跟踪器不适用）
 → 每张图打分 → 分流：animal / review / empty
```

### 为什么这样设计

| 常见做法 | 适用性 |
|---|---|
| **MegaDetector + SpeciesNet**（相机陷阱的标准流程） | 近处动物效果好。但官方文档说明它不适合"远处极小的目标"，所以这里只作为可选的近景分支。 |
| **时序背景建模** | 延时相机里减少静态背景误报的主要手段。Jenkins 等（2024）用同时段的 12 张历史图做背景，单帧检测器的 mAP 因此提升 24%。本项目沿用这一思路。 |
| **分块推理（SAHI / MegaDetector tiling）** | 能提高小目标召回，但对只有几个像素、对比度很低的兽群仍然不够，计算量也随分块数线性增长。 |
| **主动学习** | 正样本这么稀少时，最高效的标注方式。见 `review` 命令。 |

## 运行

```bash
pip install -r requirements.txt
python -m wildcam.cli synth  data/synth                         # 2 台相机 × 4 天，分训练集和测试集
python -m wildcam.cli fit    data/synth/train --out models/v1
python -m wildcam.cli run    data/synth/test  --model models/v1 --out runs/v1
python -m wildcam.cli eval   runs/v1 data/synth/test
python -m wildcam.cli review runs/v1 data/synth/test --out to_label/   # 导出最不确定的候选，供人工标注
python -m pytest -q tests
```

消融实验：加 `--background previous`，就是用"与上一帧相减"代替时段背景模型。

**真实数据的目录结构：** `root/<相机>/<相机>_YYYYmmdd_HHMMSS.jpg`。文件名里没有时间时，读取 EXIF 的 DateTimeOriginal。有标注时放在 `root/<相机>/labels.json`，格式为 `{图片名: [{"kind": "herd"|"near", "box": [x0,y0,x1,y1]}]}`，坐标基于缩放到 `image_size` 后的图片。

## 合成数据上的结果

RESULTS_PLACEHOLDER

合成数据只用来验证流程是否正确，**不代表真实效果**。真实相机还会遇到雾、镜头上的雨雪、热浪扰动、自动曝光跳变和 JPEG 伪影，生成器都没有模拟。

## 公开数据调研（2026-11）

没有找到同时满足"固定延时相机 + 苔原远景 + 远处兽群标注"的公开数据集。相近的有：

| 数据集 | 内容 | 差异 |
|---|---|---|
| LILA BC（Snapshot Serengeti、Caltech Camera Traps、ENA24 等） | 大量相机陷阱图片。Serengeti 约 76% 是空图；Caltech 共 24.3 万张、140 个机位 | 多为红外触发、近距离拍摄，没有远景兽群 |
| Marburg Camera Traps（LILA 的"其他数据集"列表） | 约 6 千张，开阔的北极平原 | 物种和标注情况待确认 |
| 阿拉斯加北部海岸平原相机（NSF Arctic Data Center） | 40 个站点，触发拍摄加 5 分钟间隔延时（2019、2021–2023 年夏季） | 最接近；标注情况需要查阅 |
| 驯鹿、麝牛的航拍数据（无人机，HerdNet 等） | 从空中俯视拍摄的兽群 | 视角不同 |

建议：先在阿拉斯加北部海岸平原的延时数据或 Marburg 数据上验证；真实部署时，用 `review` 命令做几轮主动学习，积累正样本。
