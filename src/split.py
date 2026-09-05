def split_data(X, y, train_ratio=0.8):
    split_index = int(X.shape[0] * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )
