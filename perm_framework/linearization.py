"""
Линеаризация BEN (переменный K_u)
===================================

ψ_u = BEN_u - (1/K_u) * Σ (c(x)-1)*ln(c(x)/(c(x)-1)) + (N-1)*ln(N/(N-1))

Входные данные: unit_col | item_id | ab_group (дедуплицировано).
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_linearized_ben(df, group_col="ab_group", unit_col="cookie_id", item_col="item_id", eps=1e-8):
    """Возвращает DataFrame: [unit_col, group_col, "psi", "ben_raw", "k_u"]"""
    results = []
    for group, gdf in df.groupby(group_col):
        n_users = gdf[unit_col].nunique()
        item_counts = gdf.groupby(item_col)[unit_col].nunique()
        q = item_counts / n_users
        global_corr = (n_users - 1) * np.log(n_users / (n_users - 1))

        for user, udf in gdf.groupby(unit_col):
            items = udf[item_col].unique()
            k_u = len(items)
            if k_u == 0:
                continue

            q_vals = q.reindex(items).fillna(eps).values
            q_vals = np.maximum(q_vals, eps)
            c_vals = item_counts.reindex(items).fillna(1).values.astype(float)

            ben_u = np.mean(-np.log(q_vals))
            item_corr = sum((c - 1) * np.log(c / (c - 1)) for c in c_vals if c > 1) / k_u
            psi = ben_u - item_corr + global_corr

            results.append({unit_col: user, group_col: group, "psi": psi, "ben_raw": ben_u, "k_u": k_u})
    return pd.DataFrame(results)


def linearized_ttest(
    lin_df,
    group_col="ab_group",
    control_label="control",
    treatment_label="treatment",
    alpha=0.05,
    verbose=True,
):
    psi_c = lin_df[lin_df[group_col] == control_label]["psi"].values
    psi_t = lin_df[lin_df[group_col] == treatment_label]["psi"].values

    delta = psi_t.mean() - psi_c.mean()
    t_stat, p_value = stats.ttest_ind(psi_t, psi_c, equal_var=False)

    se = np.sqrt(psi_c.var() / len(psi_c) + psi_t.var() / len(psi_t))
    s1, n1 = psi_c.var(), len(psi_c)
    s2, n2 = psi_t.var(), len(psi_t)
    df_ws = (s1 / n1 + s2 / n2) ** 2 / ((s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df_ws)

    r = {
        "delta": delta,
        "t_stat": t_stat,
        "p_value": p_value,
        "se": se,
        "ci_lower": delta - t_crit * se,
        "ci_upper": delta + t_crit * se,
        "n_control": len(psi_c),
        "n_treatment": len(psi_t),
    }

    if verbose:
        print("Linearized t-test:")
        print(f"  Δ = {delta:.6f},  SE = {se:.6f},  t = {t_stat:.4f},  p = {p_value:.8f}")
        print(f"  95% CI = [{r['ci_lower']:.6f}, {r['ci_upper']:.6f}]")
        print(f"  Решение: {'ОТВЕРГАЕМ H₀' if p_value < alpha else 'НЕ ОТВЕРГАЕМ H₀'}")
    return r

