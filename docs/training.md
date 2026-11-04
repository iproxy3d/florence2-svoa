# Обучение

## Базовая модель

Используется `microsoft/Florence-2-base-ft`. Целевой текст имеет формат:

```text
subject|action|object|adjuncts
```

Task prompt:

```text
Extract the main event or communicative gesture as subject|action|object|adjuncts:
```

## Запуск

```bash
python svoa.py train \
  --output-dir ./florence2_svoa_model \
  --epochs 30 \
  --batch-size 1 \
  --gradient-accumulation 4
```

CLI передаёт дополнительные параметры в `scripts/train_florence2_svoa.py`.

## Возобновление

Обучение автоматически продолжает последний завершённый checkpoint из `last_checkpoint`. Для нового независимого эксперимента используйте другой `--output-dir` либо явно передайте `--restart`.

## Выбор checkpoint

После каждой эпохи вычисляется validation loss. `best_model` обновляется только при улучшении. Early stopping останавливает обучение, когда validation loss не улучшается заданное число эпох.

Снижение train loss при неизменном или растущем validation loss является переобучением. В текущем baseline лучшей была эпоха 8, а не последняя эпоха 15.

## Основные результаты

В каталоге эксперимента создаются:

- `best_model/` — лучший checkpoint;
- `last_checkpoint/` — состояние для продолжения;
- `loss_history.csv` — batch- и epoch-история;
- `test_predictions.jsonl` — ответы на test split;
- `test_metrics.json` — итоговые метрики;
- `training_summary.json` — параметры запуска.

График строится командой:

```bash
python scripts/plot_florence2_loss.py \
  --history ./florence2_svoa_model/loss_history.csv
```
