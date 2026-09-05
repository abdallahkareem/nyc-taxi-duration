import mlflow


def start_run():
    mlflow.set_experiment("NYC Taxi Duration")

    return mlflow.start_run()


def log_params(
    train_size,
    test_size,
):
    mlflow.log_param(
        "model_type",
        "LinearRegression",
    )

    mlflow.log_param(
        "vectorizer",
        "DictVectorizer",
    )

    mlflow.log_param(
        "split_type",
        "chronological",
    )

    mlflow.log_param(
        "train_size",
        train_size,
    )

    mlflow.log_param(
        "test_size",
        test_size,
    )


def log_metrics(rmse, mae):
    mlflow.log_metric(
        "rmse",
        rmse,
    )

    mlflow.log_metric(
        "mae",
        mae,
    )


def log_artifact(
    path,
    artifact_path,
):
    mlflow.log_artifact(
        path,
        artifact_path=artifact_path,
    )
