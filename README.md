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

