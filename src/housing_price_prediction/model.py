from catboost import CatBoostRegressor


def build_model(
    random_state: int = 42,
    cat_features: list[str] | None = None,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        subsample=0.8,
        rsm=0.8,
        min_data_in_leaf=20,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_state,
        verbose=False,
        cat_features=cat_features or [],
    )
