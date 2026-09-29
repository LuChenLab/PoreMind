# 本地 Web UI

[English](../local-ui.md)

在已安装 PoreMind 的环境中启动 Gradio 应用：

```bash
poremind-ui
# 或在源码目录中运行
python -m ui.app
```

应用会自动打开浏览器。若页面未自动打开，可访问 `http://127.0.0.1:7860/`。

UI 分为九个步骤：

1. **Import** — 上传 ABF/CSV 文件并设置样本分组。
2. **Preprocess** — 降噪并查看信号。
3. **Pre-Events** — 在短时间窗口内调试事件检测参数。
4. **Events** — 对完整信号执行事件检测。
5. **Features & Filter** — 计算事件特征并筛选低质量事件。
6. **Reduction** — 查看 PCA、t-SNE 或 UMAP 降维结果。
7. **Train Model** — 训练传统 ML 或 DL 模型。
8. **Predict** — 加载已保存模型并分类新样本。
9. **Export** — 保存所选分析结果和参数。

<p align="center"><img src="../_generated/poremind_ui_demo.png" alt="PoreMind 本地 Web UI" width="90%"></p>

也可观看 [PoreMind 本地 Web UI 的 YouTube 演示](https://youtu.be/kSs1sbFrdPc)。
