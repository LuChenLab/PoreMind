# Local Web UI

Start the Gradio application from the installed project environment:

```bash
poremind-ui
# Or run from the source tree:
python -m ui.app
```

The UI opens a browser automatically. If it does not, visit `http://127.0.0.1:7860/`.

The interface has nine steps:

1. **Import** — load ABF/CSV files and assign sample annotations.
2. **Preprocess** — denoise and inspect signals.
3. **Pre-Events** — tune detection parameters on a short interval.
4. **Events** — run full-trace event detection.
5. **Features & Filter** — calculate event features and filter low-quality events.
6. **Reduction** — view PCA, t-SNE, or UMAP projections.
7. **Train Model** — train classical or DL models.
8. **Predict** — load a saved model and classify new samples.
9. **Export** — save selected results and analysis settings.

<p align="center"><img src="_generated/poremind_ui_demo.png" alt="PoreMind local Web UI" width="90%"></p>

See the local Web UI [video demonstration on YouTube](https://youtu.be/kSs1sbFrdPc).
