from optuna.trial import Trial


def random_forest_space(trial: Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

def xgboost_space(trial: Trial) -> dict:
    return {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400)
    }

def logistic_regression_space(trial: Trial) -> dict:
    return {
        "max_iter": 1000,
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

def svm_space(trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
    config = {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "kernel": kernel,
        "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True) if kernel in ("rbf", "poly") else "scale",
    }
    if kernel == "poly":
        config["degree"] = trial.suggest_int("degree", 2, 5)
    return config

def knn_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2),  # 1=Manhattan, 2=Euclidean
        "n_jobs": 1,
    }

def pytorch_mlp_space(trial: Trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dims = tuple(
        trial.suggest_int(f"hidden_dim_{i}", 16, 128, step=16) for i in range(n_layers)
    )
    return {
        "hidden_dims": hidden_dims,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": 30,
        "patience": 5,
    }

SEARCH_SPACES = {
    "random_forest": random_forest_space,
    "xgboost": xgboost_space,
    "logistic_regression": logistic_regression_space,
    "svm": svm_space,
    "knn": knn_space,
    "pytorch_mlp": pytorch_mlp_space
    # svm, knn similarly...
}