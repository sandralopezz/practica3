def predict_lr(data, sc, model):
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])

    X_std = sc.transform([[petal_length, petal_width]])
    # predecir flor
    y_pred = model.predict(X_std)[0]
    # predecir probabilidad 
    probability_pred = model.predict_proba(X_std)[0]

    return y_pred, probability_pred[y_pred]

def predict_svm(data, sc, model):
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])

    X_std = sc.transform([[petal_length, petal_width]])
    y_pred = model.predict(X_std)[0]
    probability_pred = model.predict_proba(X_std)[0]

    return y_pred, probability_pred[y_pred]

def predict_dt(data, sc, model):
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])

    X_std = sc.transform([[petal_length, petal_width]])
    y_pred = model.predict(X_std)[0]
    probability_pred = model.predict_proba(X_std)[0]

    return y_pred, probability_pred[y_pred]

def predict_knn(data, sc, model):
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])

    X_std = sc.transform([[petal_length, petal_width]])
    y_pred = model.predict(X_std)[0]
    probability_pred = model.predict_proba(X_std)[0]

    return y_pred, probability_pred[y_pred]
