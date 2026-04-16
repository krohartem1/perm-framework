"""
Testing Framework: Permutation Test + Bootstrap
=================================================

Входной DataFrame: unit_col | item_id | ab_group
(опционально position для BEN_weighted)

Одна строка = пользователь увидел товар (дедуплицировано).
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import time


class MetricBase(ABC):
    """User-level метрика. Возвращает [unit_col, group_col, "metric_value"]."""

    @abstractmethod
    def compute_user_values(self, df, group_col, unit_col) -> pd.DataFrame: ...

    def compute_group_delta(self, uv_df, group_col, control_label, treatment_label):
        means = uv_df.groupby(group_col)["metric_value"].mean()
        return means[treatment_label] - means[control_label]

    @property
    def name(self):
        return self.__class__.__name__


class PrecomputedMetric(MetricBase):
    """
    Метрика, которая уже посчитана на user-level.

    Ожидает DataFrame с колонками:
      - unit_col (например cookie_id)
      - group_col (например ab_group)
      - metric_col (например ben или ben_weighted)

    Возвращает [unit_col, group_col, metric_value].
    """

    def __init__(self, metric_col: str):
        self.metric_col = metric_col

    @property
    def name(self):
        return f"Precomputed({self.metric_col})"

    def compute_user_values(self, df, group_col, unit_col) -> pd.DataFrame:
        need = {unit_col, group_col, self.metric_col}
        missing = sorted(need - set(df.columns))
        if missing:
            raise ValueError(f"PrecomputedMetric ожидает колонки {sorted(need)}; не хватает: {missing}")
        uv = df[[unit_col, group_col, self.metric_col]].copy()
        uv = uv.rename(columns={self.metric_col: "metric_value"})
        return uv


class GroupLevelMetric(ABC):
    """Group-level метрика (напр. unique items). Возвращает скаляр."""

    @abstractmethod
    def compute_group_stat(self, group_df) -> float: ...

    def compute_delta(self, df, group_col, control_label, treatment_label):
        return self.compute_group_stat(df[df[group_col] == treatment_label]) - self.compute_group_stat(
            df[df[group_col] == control_label]
        )

    @property
    def name(self):
        return self.__class__.__name__


class ExperimentConfig:
    def __init__(
        self,
        unit_col="cookie_id",
        group_col="ab_group",
        item_col="item_id",
        control_label="control",
        treatment_label="treatment",
        strata_cols=None,
        n_permutations=5000,
        n_bootstrap=5000,
        alpha=0.05,
        seed=42,
    ):
        self.unit_col = unit_col
        self.group_col = group_col
        self.item_col = item_col
        self.control_label = control_label
        self.treatment_label = treatment_label
        self.strata_cols = strata_cols or []
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.seed = seed


def _permute_labels(user_info, config, rng):
    groups = np.array(user_info[config.group_col].values, dtype=object).copy()
    if not config.strata_cols:
        rng.shuffle(groups)
    else:
        for _, idx in user_info.groupby(config.strata_cols).groups.items():
            sub = groups[idx.values]
            rng.shuffle(sub)
            groups[idx.values] = sub
    return groups


def run_permutation_test(df, metric: MetricBase, config: ExperimentConfig, verbose=True) -> dict:
    c = config
    rng = np.random.default_rng(c.seed)

    uv_obs = metric.compute_user_values(df, c.group_col, c.unit_col)
    delta_obs = metric.compute_group_delta(uv_obs, c.group_col, c.control_label, c.treatment_label)
    group_stats = uv_obs.groupby(c.group_col)["metric_value"].agg(["mean", "std", "count"])

    if verbose:
        print(f"Метрика: {metric.name}")
        for grp in [c.control_label, c.treatment_label]:
            if grp in group_stats.index:
                r = group_stats.loc[grp]
                print(f"  {grp}: mean={r['mean']:.6f}, std={r['std']:.6f}, n={int(r['count'])}")
        print(f"  Δ_obs = {delta_obs:.6f}")
        print(f"  Permutation ({c.n_permutations})...")

    user_info = df[[c.unit_col, c.group_col] + c.strata_cols].drop_duplicates(subset=[c.unit_col]).reset_index(drop=True)

    deltas_perm = np.empty(c.n_permutations)
    t0 = time.time()

    for b in range(c.n_permutations):
        if verbose and (b + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {b+1}/{c.n_permutations}  (ETA: {el/(b+1)*(c.n_permutations-b-1):.0f}s)")

        pg = _permute_labels(user_info, c, rng)
        u2g = dict(zip(user_info[c.unit_col].values, pg))
        df_p = df.copy()
        df_p[c.group_col] = df_p[c.unit_col].map(u2g)
        uv_p = metric.compute_user_values(df_p, c.group_col, c.unit_col)
        deltas_perm[b] = metric.compute_group_delta(uv_p, c.group_col, c.control_label, c.treatment_label)

    p_abs = (np.sum(np.abs(deltas_perm) >= np.abs(delta_obs)) + 1) / (c.n_permutations + 1)
    p_r = (np.sum(deltas_perm >= delta_obs) + 1) / (c.n_permutations + 1)
    p_l = (np.sum(deltas_perm <= delta_obs) + 1) / (c.n_permutations + 1)
    p_2s = min(2 * min(p_r, p_l), 1.0)
    elapsed = time.time() - t0

    if verbose:
        print(f"  p-value (|Δ|) = {p_abs:.6f},  p-value (2s) = {p_2s:.6f}")
        print(f"  Решение: {'ОТВЕРГАЕМ H₀' if p_abs < c.alpha else 'НЕ ОТВЕРГАЕМ H₀'} ({elapsed:.0f}s)")

    return {
        "metric_name": metric.name,
        "delta_obs": delta_obs,
        "deltas_perm": deltas_perm,
        "p_value_abs": p_abs,
        "p_value_2sided": p_2s,
        "group_stats": group_stats,
        "user_values_obs": uv_obs,
        "n_permutations": c.n_permutations,
        "alpha": c.alpha,
        "elapsed_seconds": elapsed,
    }


def run_bootstrap_test(df, metric: GroupLevelMetric, config: ExperimentConfig, verbose=True) -> dict:
    """
    Bootstrap CI для group-level метрик (generic).

    Для `UniqueItems` на больших raw-таблицах (десятки миллионов строк) обычно быстрее
    использовать `run_bootstrap_unique_items()` — он избегает `isin`-фильтрации на каждой итерации.
    """
    c = config
    rng = np.random.default_rng(c.seed)

    delta_obs = metric.compute_delta(df, c.group_col, c.control_label, c.treatment_label)
    stat_c = metric.compute_group_stat(df[df[c.group_col] == c.control_label])
    stat_t = metric.compute_group_stat(df[df[c.group_col] == c.treatment_label])

    if verbose:
        print(f"Метрика: {metric.name}")
        print(f"  {c.control_label}: {stat_c:.0f},  {c.treatment_label}: {stat_t:.0f},  Δ = {delta_obs:.0f}")
        print(f"  Bootstrap ({c.n_bootstrap})...")

    users_by_grp = {grp: df[df[c.group_col] == grp][c.unit_col].unique() for grp in [c.control_label, c.treatment_label]}

    deltas_boot = np.empty(c.n_bootstrap)
    t0 = time.time()

    for b in range(c.n_bootstrap):
        if verbose and (b + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"  {b+1}/{c.n_bootstrap}  (ETA: {el/(b+1)*(c.n_bootstrap-b-1):.0f}s)")

        stats_b = {}
        for grp in [c.control_label, c.treatment_label]:
            users = users_by_grp[grp]
            sampled = np.unique(rng.choice(users, size=len(users), replace=True))
            grp_df = df[(df[c.group_col] == grp) & (df[c.unit_col].isin(sampled))]
            stats_b[grp] = metric.compute_group_stat(grp_df)
        deltas_boot[b] = stats_b[c.treatment_label] - stats_b[c.control_label]

    ci_lo = np.percentile(deltas_boot, 100 * c.alpha / 2)
    ci_hi = np.percentile(deltas_boot, 100 * (1 - c.alpha / 2))
    sig = (ci_lo > 0) or (ci_hi < 0)
    p_approx = min(
        2 * (np.sum(deltas_boot <= 0 if delta_obs > 0 else deltas_boot >= 0) + 1) / (c.n_bootstrap + 1),
        1.0,
    )
    elapsed = time.time() - t0

    if verbose:
        print(f"  95% CI = [{ci_lo:.0f}, {ci_hi:.0f}],  p ≈ {p_approx:.6f}")
        print(f"  Решение: {'ОТВЕРГАЕМ H₀' if sig else 'НЕ ОТВЕРГАЕМ H₀'} ({elapsed:.0f}s)")

    return {
        "metric_name": metric.name,
        "delta_obs": delta_obs,
        "deltas_boot": deltas_boot,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_value_approx": p_approx,
        "significant": sig,
        "stat_control": stat_c,
        "stat_treatment": stat_t,
        "n_bootstrap": c.n_bootstrap,
        "alpha": c.alpha,
        "elapsed_seconds": elapsed,
    }


def run_bootstrap_unique_items(df, config: ExperimentConfig, item_col="item_id", verbose=True) -> dict:
    """
    Ускоренный bootstrap для числа уникальных товаров (UniqueItems).

    Вместо `isin`-фильтрации на каждой итерации (O(N_rows)):
    1) Один раз строим dict user → массив item_codes (int32).
    2) На каждой итерации: семплируем юзеров, concat их item_codes,
       ставим bitmap[item_code] = True, считаем sum.

    Важно: `df` должен быть дедуплирован по (unit_col, item_col) за период (как в гайде).
    """
    c = config
    rng = np.random.default_rng(c.seed)

    if verbose:
        print("UniqueItems bootstrap (ускоренный)")
        print("  Подготовка...")

    t0 = time.time()

    # Кодируем item_id в int32 0..n_items-1
    item_codes_arr, item_uniques = pd.factorize(df[item_col], sort=False)
    item_codes_arr = item_codes_arr.astype(np.int32)
    n_items_enc = len(item_uniques)

    # Строим dict user → массив item_codes для каждой группы
    user_items_by_grp = {}
    users_by_grp = {}
    for grp in [c.control_label, c.treatment_label]:
        mask = df[c.group_col].values == grp
        gdf_users = df[c.unit_col].values[mask]
        gdf_codes = item_codes_arr[mask]

        order = np.argsort(gdf_users, kind="stable")
        users_sorted = gdf_users[order]
        items_sorted = gdf_codes[order]

        boundaries = np.concatenate([[0], np.where(np.diff(users_sorted) != 0)[0] + 1, [len(users_sorted)]])
        unique_users = users_sorted[boundaries[:-1]]

        user_items = {}
        for j in range(len(unique_users)):
            user_items[unique_users[j]] = items_sorted[boundaries[j] : boundaries[j + 1]]

        user_items_by_grp[grp] = user_items
        users_by_grp[grp] = unique_users

    t_prep = time.time() - t0

    # Наблюдаемые значения
    stat_c = int(np.unique(item_codes_arr[df[c.group_col].values == c.control_label]).size)
    stat_t = int(np.unique(item_codes_arr[df[c.group_col].values == c.treatment_label]).size)
    delta_obs = stat_t - stat_c

    if verbose:
        print(f"  Подготовка: {t_prep:.1f}s, bitmap: {n_items_enc * 1 / 1024 / 1024:.1f} MB")
        print(f"  {c.control_label}: {stat_c:,},  {c.treatment_label}: {stat_t:,},  Δ = {delta_obs:,}")
        print(f"  Bootstrap ({c.n_bootstrap})...")

    deltas_boot = np.empty(c.n_bootstrap)
    t0 = time.time()

    for b in range(c.n_bootstrap):
        if verbose and (b + 1) % 500 == 0:
            el = time.time() - t0
            eta = el / (b + 1) * (c.n_bootstrap - b - 1)
            print(f"  {b+1}/{c.n_bootstrap}  (ETA: {eta:.0f}s)")

        stats_b = {}
        for grp in [c.control_label, c.treatment_label]:
            users = users_by_grp[grp]
            user_items = user_items_by_grp[grp]

            sampled = np.unique(rng.choice(users, size=len(users), replace=True))

            all_items = np.concatenate([user_items[u] for u in sampled])

            bitmap = np.zeros(n_items_enc, dtype=bool)
            bitmap[all_items] = True

            stats_b[grp] = int(bitmap.sum())

        deltas_boot[b] = stats_b[c.treatment_label] - stats_b[c.control_label]

    ci_lo = np.percentile(deltas_boot, 100 * c.alpha / 2)
    ci_hi = np.percentile(deltas_boot, 100 * (1 - c.alpha / 2))
    sig = (ci_lo > 0) or (ci_hi < 0)
    if delta_obs > 0:
        p_approx = 2 * (np.sum(deltas_boot <= 0) + 1) / (c.n_bootstrap + 1)
    else:
        p_approx = 2 * (np.sum(deltas_boot >= 0) + 1) / (c.n_bootstrap + 1)
    p_approx = min(p_approx, 1.0)
    elapsed = time.time() - t0

    if verbose:
        print(f"  95% CI = [{ci_lo:.0f}, {ci_hi:.0f}],  p ≈ {p_approx:.6f}")
        print(f"  Решение: {'ОТВЕРГАЕМ H₀' if sig else 'НЕ ОТВЕРГАЕМ H₀'} ({elapsed:.0f}s)")

    return {
        "metric_name": "UniqueItems",
        "delta_obs": delta_obs,
        "deltas_boot": deltas_boot,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_value_approx": p_approx,
        "significant": sig,
        "stat_control": stat_c,
        "stat_treatment": stat_t,
        "n_bootstrap": c.n_bootstrap,
        "alpha": c.alpha,
        "elapsed_seconds": elapsed,
    }


def plot_permutation_results(results, save_path=None):
    d = results["deltas_perm"]
    obs = results["delta_obs"]
    p = results["p_value_abs"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(d, bins=60, color="#4a90d9", alpha=0.7, edgecolor="white")
    ax.axvline(obs, color="#e74c3c", lw=2.5, label=f"Δ_obs={obs:.5f}")
    ax.axvline(-abs(obs), color="#e74c3c", lw=2.5, ls="--", alpha=0.5)
    ax.set_xlabel("Δ")
    ax.set_ylabel("Частота")
    ax.set_title(f"{results['metric_name']}: null (p={p:.5f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bootstrap_results(results, save_path=None):
    d = results["deltas_boot"]
    obs = results["delta_obs"]
    lo, hi = results["ci_lower"], results["ci_upper"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(d, bins=60, color="#27ae60", alpha=0.7, edgecolor="white")
    ax.axvline(obs, color="#e74c3c", lw=2.5, label=f"Δ_obs={obs:.0f}")
    ax.axvline(lo, color="#f39c12", lw=2, ls="--", label=f"CI=[{lo:.0f}, {hi:.0f}]")
    ax.axvline(hi, color="#f39c12", lw=2, ls="--")
    ax.axvline(0, color="gray", lw=1.5, ls=":")
    ax.set_xlabel("Δ")
    ax.set_ylabel("Частота")
    ax.set_title(f"{results['metric_name']}: bootstrap")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_user_distributions(results, group_col="ab_group", save_path=None):
    uv = results["user_values_obs"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for g in sorted(uv[group_col].unique()):
        v = uv[uv[group_col] == g]["metric_value"].values
        ax.hist(v, bins=50, alpha=0.5, label=f"{g} (μ={v.mean():.4f})", edgecolor="white")
    ax.set_xlabel("metric_value")
    ax.set_ylabel("Число пользователей")
    ax.set_title(f"{results['metric_name']}: по группам")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

