"""
Валидация данных: unit_col | item_id | ab_group
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


class CheckResult:
    def __init__(self, name, status, message, details=None):
        self.name, self.status, self.message = name, status, message
        self.details = details or {}

    def __repr__(self):
        icon = {"PASS": "✓", "WARNING": "⚠", "FAIL": "✗"}[self.status]
        return f"  [{icon} {self.status}] {self.name}: {self.message}"


def check_no_cross_group(df, unit_col, group_col):
    bad = df.groupby(unit_col)[group_col].nunique()
    bad = bad[bad > 1]
    if len(bad) == 0:
        return CheckResult("Cross-group", "PASS", "Нет пользователей в нескольких группах")
    return CheckResult("Cross-group", "FAIL", f"{len(bad)} пользователей в нескольких группах!")


def check_group_sizes(df, unit_col, group_col):
    sizes = df.groupby(group_col)[unit_col].nunique()
    total = sizes.sum()
    imb = (sizes.max() - sizes.min()) / total * 100
    info = {g: f"{n} ({n/total*100:.1f}%)" for g, n in sizes.items()}
    st = "PASS" if imb <= 5 else "WARNING"
    return CheckResult("Group sizes", st, f"Дисбаланс {imb:.1f}%. {info}")


def check_duplicates(df, unit_col, item_col):
    n = len(df)
    nd = df.drop_duplicates(subset=[unit_col, item_col]).shape[0]
    d = n - nd
    if d == 0:
        return CheckResult("Duplicates", "PASS", "Нет дубликатов (user, item)")
    return CheckResult("Duplicates", "WARNING" if d / n < 0.01 else "FAIL", f"{d} дубликатов ({d/n*100:.2f}%)")


def check_items_per_user(df, unit_col, item_col, group_col):
    k = df.groupby([unit_col, group_col])[item_col].nunique().reset_index()
    k.columns = [unit_col, group_col, "k_u"]

    stats = {}
    for grp, gdf in k.groupby(group_col):
        v = gdf["k_u"].values
        stats[grp] = f"mean={v.mean():.1f}, med={np.median(v):.0f}, [{v.min()}, {v.max()}]"

    groups = list(k[group_col].unique())
    if len(groups) == 2:
        k1 = k[k[group_col] == groups[0]]["k_u"].values
        k2 = k[k[group_col] == groups[1]]["k_u"].values
        _, p = sp_stats.mannwhitneyu(k1, k2, alternative="two-sided")
    else:
        p = 1.0

    zero = (k["k_u"] == 0).sum()
    if zero > 0:
        return CheckResult("K_u distribution", "FAIL", f"{zero} пользователей с 0 товаров!")

    msg = "; ".join(f"{g}: {s}" for g, s in stats.items())
    st = "WARNING" if p < 0.01 else "PASS"
    return CheckResult("K_u distribution", st, f"MW p={p:.4f}. {msg}")


def check_item_overlap(df, item_col, group_col):
    groups = df[group_col].unique()
    if len(groups) != 2:
        return CheckResult("Item overlap", "PASS", f"{len(groups)} групп")
    s1 = set(df[df[group_col] == groups[0]][item_col].unique())
    s2 = set(df[df[group_col] == groups[1]][item_col].unique())
    j = len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0
    st = "WARNING" if j < 0.3 else "PASS"
    return CheckResult(
        "Item overlap",
        st,
        f"Jaccard={j:.3f}. {groups[0]}: {len(s1)}, {groups[1]}: {len(s2)}, общих: {len(s1&s2)}",
    )


def check_rare_items(df, unit_col, item_col, group_col):
    info = {}
    for grp, gdf in df.groupby(group_col):
        c = gdf.groupby(item_col)[unit_col].nunique()
        ns = (c == 1).sum()
        info[grp] = f"{ns}/{len(c)} ({ns/len(c)*100:.1f}%) singleton"
    msg = ", ".join(f"{g}: {v}" for g, v in info.items())
    mx = max((gdf.groupby(item_col)[unit_col].nunique() == 1).mean() for _, gdf in df.groupby(group_col)) * 100
    return CheckResult("Rare items", "WARNING" if mx > 50 else "PASS", msg)


def check_ben_outliers(df, unit_col, item_col, group_col, eps=1e-8):
    info = {}
    for grp, gdf in df.groupby(group_col):
        n = gdf[unit_col].nunique()
        reach = gdf.groupby(item_col)[unit_col].nunique() / n
        bens = []
        for _, udf in gdf.groupby(unit_col):
            items = udf[item_col].unique()
            if len(items) == 0:
                continue
            q = reach.reindex(items).fillna(eps).values
            q = np.maximum(q, eps)
            bens.append(np.mean(-np.log(q)))
        bens = np.array(bens)
        med = np.median(bens)
        mad = np.median(np.abs(bens - med))
        no = np.sum(np.abs(bens - med) > 5 * max(mad, 0.01))
        info[grp] = f"mean={bens.mean():.3f}, std={bens.std():.3f}, outliers(5MAD)={no}"
    msg = "; ".join(f"{g}: {v}" for g, v in info.items())
    return CheckResult("BEN outliers", "PASS", msg)


def run_all_checks(df, unit_col="cookie_id", group_col="ab_group", item_col="item_id", verbose=True):
    checks = [
        check_no_cross_group(df, unit_col, group_col),
        check_group_sizes(df, unit_col, group_col),
        check_duplicates(df, unit_col, item_col),
        check_items_per_user(df, unit_col, item_col, group_col),
        check_item_overlap(df, item_col, group_col),
        check_rare_items(df, unit_col, item_col, group_col),
        check_ben_outliers(df, unit_col, item_col, group_col),
    ]
    if verbose:
        np_, nw, nf = (sum(1 for c in checks if c.status == s) for s in ["PASS", "WARNING", "FAIL"])
        print("=" * 60)
        for c in checks:
            print(c)
        print(f"\nИтого: {np_} PASS, {nw} WARNING, {nf} FAIL")
        if nf:
            print("⚠ ЕСТЬ FAIL!")
        print("=" * 60)
    return checks

