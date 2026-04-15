## perm-framework (локально / в ноутбуках)

Этот репозиторий содержит пакет `perm_framework`:

- permutation test / bootstrap
- метрики `BEN`, `BEN_weighted`, `UniqueItems`
- валидации датасета
- линеаризация BEN (для ускорения)

### Требование к входному датасету (строго)

`DataFrame` должен быть **дедуплицирован за весь период**: одна строка на `(unit_col, item_id)`.

Колонки:
- `cookie_id` или `user_id` (задаёшь как `unit_col`)
- `item_id`
- `ab_group`
- опционально `position` (для `BEN_weighted`)

### User-level режим (когда BEN уже посчитан в SQL)

Если ты считаешь BEN/BEN_weighted в SQL на user-level, то для permutation test достаточно DataFrame:

- `cookie_id` (или `user_id`)
- `ab_group`
- `ben` (и/или `ben_weighted`)

Пример:

```python
from perm_framework import ExperimentConfig, run_permutation_test, PrecomputedMetric

config = ExperimentConfig(unit_col="cookie_id", group_col="ab_group", n_permutations=5000)
res_ben = run_permutation_test(df_user, PrecomputedMetric("ben"), config)
res_benw = run_permutation_test(df_user, PrecomputedMetric("ben_weighted"), config)
```

## Два режима работы (как у тебя)

### Режим 1 (default): raw impressions df (`cookie_id|item_id|ab_group|position`)

Ты можешь сначала взять небольшой сэмпл пользователей (например 5% в каждой группе),
построить для них raw df и прогнать “честный” расчёт BEN из исходных `(user,item)`.

```python
from perm_framework import (
    ExperimentConfig,
    run_permutation_test,
    BENMetric,
    BENWeightedMetric,
    sample_users_by_group,
)

df_small = sample_users_by_group(df_raw, frac=0.05, seed=42)
config = ExperimentConfig(unit_col="cookie_id", group_col="ab_group", n_permutations=2000)

res_ben  = run_permutation_test(df_small, BENMetric(), config)
res_benw = run_permutation_test(df_small, BENWeightedMetric(), config)
```

SQL (рекомендуемый способ сделать подвыборку 5% пользователей воспроизводимо):

```sql
CREATE TABLE avc_impr_sample5 AS
SELECT *
FROM avc_impressions_ab_30859
WHERE (abs(xxhash64(CAST(cookie_id AS varchar))) % 100) < 5;
```

### Режим 2: precomputed user-level df (`cookie_id|ab_group|ben|ben_weighted|k_u`)

Ты считаешь метрики в SQL на user-level, а в Python делаешь только permutation test.

```python
from perm_framework import ExperimentConfig, run_permutation_test, PrecomputedMetric

config = ExperimentConfig(unit_col="cookie_id", group_col="ab_group", n_permutations=5000)
res_ben  = run_permutation_test(df_user, PrecomputedMetric("ben"), config)
res_benw = run_permutation_test(df_user, PrecomputedMetric("ben_weighted"), config)
```

Валидация user-level df:

```python
from perm_framework import run_user_level_checks
run_user_level_checks(df_user, metric_cols=("ben","ben_weighted","k_u"))
```

### Приближённая user-level “уникальные айтемы”

Group-level `UniqueItems` (мощность объединения) на бутстрапе по сырым данным очень тяжёлая.
Для быстрой прикидки можно использовать user-level метрику `K_u` = число уникальных айтемов у пользователя:

```python
from perm_framework import UserUniqueItemsMetric
res_ku = run_permutation_test(df_raw, UserUniqueItemsMetric(), config)
```

### Как подключить в ноутбуке (вариант 1: clone + editable install)

```bash
git clone <твой-репозиторий>
cd <папка-репозитория>
python -m pip install -e .
```

В ноутбуке:

```python
from perm_framework import (
    ExperimentConfig,
    run_permutation_test,
    run_bootstrap_test,
    BENMetric,
    BENWeightedMetric,
    UniqueItemsMetric,
    run_all_checks,
)
```

### Как подключить в ноутбуке (вариант 2: без установки, через sys.path)

```python
import sys
sys.path.insert(0, "/path/to/repo")

from perm_framework import ExperimentConfig, run_permutation_test, BENMetric
```

