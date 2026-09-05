from sklearn.feature_extraction import DictVectorizer


def vectorize_features(df):
    X = df[
        [
            "PU_DO",
            "trip_distance",
        ]
    ]

    X_dicts = X.to_dict(orient="records")

    dv = DictVectorizer()

    X_vec = dv.fit_transform(X_dicts)

    return X_vec, dv
