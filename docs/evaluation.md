# Оценка качества

## Основные метрики

- `valid_format` — доля ответов с четырьмя полями;
- `exact_match` — полное совпадение всех четырёх полей;
- `subject accuracy`;
- `action accuracy`;
- `object accuracy`;
- `adjuncts accuracy`.

Полное совпадение является строгой метрикой: ошибка в одном поле делает весь пример несовпавшим. Поэтому её нужно рассматривать вместе с точностью отдельных ролей.

## Текущий baseline

| Метрика | Значение |
|---|---:|
| Примеры | 2 231 |
| `valid_format` | 0.9951 |
| `exact_match` | 0.2766 |
| `subject` | 0.7768 |
| `action` | 0.6333 |
| `object` | 0.6365 |
| `adjuncts` | 0.5701 |

Лучший validation loss: `0.45876`, эпоха 8. Эти значения фиксируют текущую исходную точку и не являются окончательной оценкой после полной переработки датасета SWiG/HaGRID.

## Повторный расчёт

```bash
python scripts/evaluate_predictions.py \
  --predictions ./florence2_svoa_model/test_predictions.jsonl \
  --output ./florence2_svoa_model/test_metrics_recalculated.json
```

## Что ещё требуется для финальной публикации

- macro/micro F1 по каждому полю;
- частичное совпадение события;
- частота `none` и `unknown` по ролям;
- показатели по источникам и типам сцен;
- confusion analysis для действий и жестов;
- ручная экспертная проверка выборки;
- сравнение с VLM до дообучения.
