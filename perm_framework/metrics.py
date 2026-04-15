"""
Метрики: BEN + UniqueItems
===========================

Входные данные: unit_col | item_id | ab_group
(дедуплицировано по unit_col + item_id)
"""

import numpy as np
import pandas as pd

from .framework import MetricBase, GroupLevelMetric


class BENMetric(MetricBase):
    """
    BEN_u = (1/K_u) * Σ -log(q(x))
    K_u = число уникальных товаров у пользователя (вся выдача).
    q(x) = доля пользователей в группе, увидевших товар x хотя бы раз.
    """

    def __init__(self, item_col="item_id", eps=1e-8):
        self.item_col = item_col
        self.eps = eps

    @property
    def name(self):
        return "BEN"

    def compute_user_values(self, df, group_col, unit_col):
        results = []
        for group, gdf in df.groupby(group_col):
            n_users = gdf[unit_col].nunique()
            reach = gdf.groupby(self.item_col)[unit_col].nunique() / n_users

            for user, udf in gdf.groupby(unit_col):
                items = udf[self.item_col].unique()
                if len(items) == 0:
                    continue
                q = reach.reindex(items).fillna(self.eps).values
                q = np.maximum(q, self.eps)
                results.append({unit_col: user, group_col: group, "metric_value": np.mean(-np.log(q))})
        return pd.DataFrame(results)


class BENWeightedMetric(MetricBase):
    """BEN с позиционным взвешиванием. Требует колонку position."""

    def __init__(self, item_col="item_id", position_col="position", eps=1e-8):
        self.item_col = item_col
        self.position_col = position_col
        self.eps = eps

    @property
    def name(self):
        return "BEN_weighted"

    def compute_user_values(self, df, group_col, unit_col):
        results = []
        for group, gdf in df.groupby(group_col):
            n_users = gdf[unit_col].nunique()
            reach = gdf.groupby(self.item_col)[unit_col].nunique() / n_users

            for user, udf in gdf.groupby(unit_col):
                dedup = udf.sort_values(self.position_col).drop_duplicates(subset=[self.item_col], keep="first")
                items = dedup[self.item_col].values
                pos = dedup[self.position_col].values
                if len(items) == 0:
                    continue
                q = reach.reindex(items).fillna(self.eps).values
                q = np.maximum(q, self.eps)
                w = 1.0 / np.log2(pos + 1)
                results.append(
                    {unit_col: user, group_col: group, "metric_value": np.sum(-np.log(q) * w) / w.sum()}
                )
        return pd.DataFrame(results)


class UniqueItemsMetric(GroupLevelMetric):
    """Число уникальных товаров в группе. Для bootstrap."""

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
    Это НЕ то же самое, что group-level UniqueItems (мощность объединения),
    но полезно как приближённая оценка и легко считается/тестируется.
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

        uv = (
            df.groupby([unit_col, group_col])[self.item_col]
            .nunique()
            .reset_index()
            .rename(columns={self.item_col: "metric_value"})
        )
        return uv

