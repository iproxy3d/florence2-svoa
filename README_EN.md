# Florence-2 SVOA

**Structured visual event extraction from images and keyframes.**

[Русская версия](README.md)

The project fine-tunes Florence-2 to produce one directly observable event in a deterministic four-field format:

```text
subject|action|object|adjuncts
```

Example:

```text
man|smile|none|skatepark
```

This is not an emotion classifier or a motion-dynamics model. It answers **what is happening in the frame**. Video segmentation, frame selection, temporal aggregation and visual-stimulus strength are separate layers.

## Four-command workflow

```bash
python -m pip install -e ".[all]"
python svoa.py download --limit-per-dataset 5000
python svoa.py prepare --env-file .env --limit-per-dataset 5000
python svoa.py train --output-dir ./florence2_svoa_model
python svoa.py infer --model ./florence2_svoa_model/best_model --image ./example.jpg
```

The `prepare` command runs VLM labelling and the official SWiG/HaGRID corrections internally. The final training dataset is written to `svoa_work_v2/dataset.jsonl`.

## Current experimental baseline

| Metric | Value |
|---|---:|
| Test examples | 2,231 |
| Valid format | 99.51% |
| Exact four-field match | 27.66% |
| Subject accuracy | 77.68% |
| Action accuracy | 63.33% |
| Object accuracy | 63.65% |
| Adjuncts accuracy | 57.01% |

Best validation loss was `0.45876` at epoch 8. See [evaluation details](docs/evaluation.md) and [limitations](docs/limitations.md).

## License

Project code is released under the [MIT License](LICENSE). The base model and source datasets retain their own licenses and usage terms.
