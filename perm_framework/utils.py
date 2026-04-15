import pandas as pd


def sample_users_by_group(
    df: pd.DataFrame,
    *,
    unit_col: str = "cookie_id",
    group_col: str = "ab_group",
    frac: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Семплирует пользователей внутри каждой группы (stratified by group),
    затем возвращает строки исходного df только для выбранных пользователей.
    """
    if not (0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1].")
    need = {unit_col, group_col}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"sample_users_by_group ожидает колонки {sorted(need)}; не хватает: {missing}")

    users = df[[unit_col, group_col]].drop_duplicates(subset=[unit_col])
    sampled_parts = []
    for _, g in users.groupby(group_col, sort=False):
        sampled_parts.append(g.sample(frac=frac, random_state=seed))
    sampled_users = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else users.head(0)
    keep = set(sampled_users[unit_col].tolist())
    return df[df[unit_col].isin(keep)].copy()

