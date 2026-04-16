"""
Метрики (ускоренная версия через np.bincount)
==============================================

RAW MODE (вход: cookie_id | item_id | ab_group | position):
  - BENMetric            — BEN
  - BENWeightedMetric    — BEN с позиционным взвешиванием
  - UniqueItemsMetric    — число уникальных товаров в группе

PRECOMPUTED MODE (вход: cookie_id | ab_group | ben | ben_weighted | k_u):
  Используй PrecomputedMetric из framework.py.

Реализация через np.bincount даёт ~15-50x ускорение по сравнению с циклами/merge.
API полностью совместим с MetricBase — работает через обычный run_permutation_test.
"""

import numpy as np
import pandas as pd

from .framework import MetricBase, GroupLevelMetric


# ============================================================
# USER-LEVEL МЕТРИКИ
# ============================================================


def _encode_ids(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Переводит любой идентификатор (bigint или строка) в int32 0..N-1.
    Возвращает (коды, уникальные значения).
    Для bigint работает так же быстро, как для строк — pandas factorize векторизован.
    """
    codes, uniques = pd.factorize(series, sort=False)
    return codes.astype(np.int32), uniques


class BENMetric(MetricBase):
    """
    BEN_u = (1/K_u) * Σ -log(q(x))

    q(x) = доля юзеров в группе, увидевших товар x.
    K_u = число уникальных товаров у юзера.

    Реализация через np.bincount:
    1. Кодируем cookie_id, item_id в int 0..N-1.
    2. Для каждой группы считаем reach через bincount(items_in_group).
    3. Для каждой строки df присваиваем -log q(item) из её группы.
    4. BEN_u = sum(-log q) / K_u через bincount по юзерам.

    Никаких циклов по юзерам, никаких merge по string-ключам.
    """

    def __init__(self, item_col="item_id", eps=1e-8):
        self.item_col = item_col
        self.eps = eps

    @property
    def name(self):
        return "BEN"

    def compute_user_values(self, df, group_col, unit_col):
        # 1) Кодируем ID в int
        u_codes, u_uniques = _encode_ids(df[unit_col])
        i_codes, _ = _encode_ids(df[self.item_col])
        g_codes, g_uniques = _encode_ids(df[group_col])

        n_users = len(u_uniques)
        n_items = int(i_codes.max()) + 1 if len(i_codes) else 0
        n_groups = len(g_uniques)

        # 2) Группа каждого юзера (первая встреча при стабильной сортировке)
        order = np.argsort(u_codes, kind="stable")
        u_sorted = u_codes[order]
        g_sorted = g_codes[order]
        first_idx = np.concatenate([[0], np.where(np.diff(u_sorted) != 0)[0] + 1])
        user_group_vec = np.empty(n_users, dtype=np.int32)
        user_group_vec[u_sorted[first_idx]] = g_sorted[first_idx]

        # 3) Дедупликация (u, i) пар
        pair_keys = u_codes.astype(np.int64) * n_items + i_codes.astype(np.int64)
        unique_pair_idx = np.unique(pair_keys, return_index=True)[1]
        u_dedup = u_codes[unique_pair_idx]
        i_dedup = i_codes[unique_pair_idx]
        g_dedup = user_group_vec[u_dedup]

        k_u = np.bincount(u_dedup, minlength=n_users).astype(np.float64)
        group_sizes = np.bincount(user_group_vec, minlength=n_groups).astype(np.float64)

        # 4) Reach и -log q per group, через bincount
        neg_log_q_per_row = np.empty(len(u_dedup), dtype=np.float64)
        for gi in range(n_groups):
            mask = g_dedup == gi
            items_g = i_dedup[mask]
            c_x = np.bincount(items_g, minlength=n_items).astype(np.float64)
            q_x = np.maximum(c_x / max(group_sizes[gi], 1), self.eps)
            nlq = -np.log(q_x)
            neg_log_q_per_row[mask] = nlq[i_dedup[mask]]

        # 5) BEN_u = sum(-log q) / k_u
        sum_nlq = np.bincount(u_dedup, weights=neg_log_q_per_row, minlength=n_users)
        ben_u = np.zeros(n_users, dtype=np.float64)
        nonzero = k_u > 0
        ben_u[nonzero] = sum_nlq[nonzero] / k_u[nonzero]

        # 6) Возвращаем DataFrame
        return pd.DataFrame(
            {
                unit_col: u_uniques[nonzero],
                group_col: g_uniques[user_group_vec[nonzero]],
                "metric_value": ben_u[nonzero],
            }
        )


class BENWeightedMetric(MetricBase):
    """
    BEN_weighted_u = Σ w_r * (-log q(x)) / Σ w_r,   w_r = 1/log2(pos + 1)

    Реализация через np.bincount с weighted sum.
    """

    def __init__(self, item_col="item_id", position_col="position", eps=1e-8):
        self.item_col = item_col
        self.position_col = position_col
        self.eps = eps

    @property
    def name(self):
        return "BEN_weighted"

    def compute_user_values(self, df, group_col, unit_col):
        # Кодирование
        u_codes, u_uniques = _encode_ids(df[unit_col])
        i_codes, _ = _encode_ids(df[self.item_col])
        g_codes, g_uniques = _encode_ids(df[group_col])
        pos = df[self.position_col].values

        n_users = len(u_uniques)
        n_items = int(i_codes.max()) + 1 if len(i_codes) else 0
        n_groups = len(g_uniques)

        # Дедупликация (u, i) с минимальной позицией
        # Сортируем по (u, i, pos), берём первую встречу каждой пары
        order = np.lexsort((pos, i_codes, u_codes))
        u_sorted = u_codes[order]
        i_sorted = i_codes[order]
        g_sorted = g_codes[order]
        pos_sorted = pos[order]

        # Первая строка для каждой уникальной пары (u, i)
        pair_keys = u_sorted.astype(np.int64) * n_items + i_sorted.astype(np.int64)
        first_mask = np.concatenate([[True], np.diff(pair_keys) != 0])
        u_dedup = u_sorted[first_mask]
        i_dedup = i_sorted[first_mask]
        g_dedup = g_sorted[first_mask]
        pos_dedup = pos_sorted[first_mask]

        # Группа каждого юзера (первая встреча)
        order_u = np.argsort(u_dedup, kind="stable")
        u_sorted2 = u_dedup[order_u]
        g_sorted2 = g_dedup[order_u]
        first_idx = np.concatenate([[0], np.where(np.diff(u_sorted2) != 0)[0] + 1])
        user_group_vec = np.empty(n_users, dtype=np.int32)
        user_group_vec[u_sorted2[first_idx]] = g_sorted2[first_idx]

        group_sizes = np.bincount(user_group_vec, minlength=n_groups).astype(np.float64)

        # Reach и -log q
        neg_log_q_per_row = np.empty(len(u_dedup), dtype=np.float64)
        for gi in range(n_groups):
            mask = g_dedup == gi
            items_g = i_dedup[mask]
            c_x = np.bincount(items_g, minlength=n_items).astype(np.float64)
            q_x = np.maximum(c_x / max(group_sizes[gi], 1), self.eps)
            nlq = -np.log(q_x)
            neg_log_q_per_row[mask] = nlq[i_dedup[mask]]

        # Веса
        w = 1.0 / np.log2(pos_dedup.astype(np.float64) + 1)

        # Взвешенные суммы на юзера
        sum_wnlq = np.bincount(u_dedup, weights=neg_log_q_per_row * w, minlength=n_users)
        sum_w = np.bincount(u_dedup, weights=w, minlength=n_users)

        ben_w = np.zeros(n_users, dtype=np.float64)
        nonzero = sum_w > 0
        ben_w[nonzero] = sum_wnlq[nonzero] / sum_w[nonzero]

        return pd.DataFrame(
            {
                unit_col: u_uniques[nonzero],
                group_col: g_uniques[user_group_vec[nonzero]],
                "metric_value": ben_w[nonzero],
            }
        )


# ============================================================
# GROUP-LEVEL МЕТРИКИ
# ============================================================


class UniqueItemsMetric(GroupLevelMetric):
    """Число уникальных товаров в группе (group-level, для bootstrap)."""

    def __init__(self, item_col="item_id"):
        self.item_col = item_col

    @property
    def name(self):
        return "UniqueItems"

    def compute_group_stat(self, group_df):
        return group_df[self.item_col].nunique()


class UserUniqueItemsMetric(MetricBase):
    """
    User-level метрика: число уникальных айтемов у пользователя (K_u).
    Это не то же самое, что group-level UniqueItems, но удобно как приближение.
    """

    def __init__(self, item_col="item_id"):
        self.item_col = item_col

    @property
    def name(self):
        return "UserUniqueItems"

    def compute_user_values(self, df, group_col, unit_col):
        need = {unit_col, group_col, self.item_col}
        missing = sorted(need - set(df.columns))
        if missing:
            raise ValueError(f"UserUniqueItemsMetric ожидает колонки {sorted(need)}; не хватает: {missing}")

        return (
            df.groupby([unit_col, group_col])[self.item_col]
            .nunique()
            .reset_index()
            .rename(columns={self.item_col: "metric_value"})
        )
