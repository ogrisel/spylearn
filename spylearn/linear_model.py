"""Parallel Linear Model training with partial_fit and averaging"""

from sklearn.base import clone


def _train(iterator, model, classes, skip=None):
    for X, y in iterator:
        model.partial_fit(X, y, classes=classes)
    yield model, 1


def _model_sum(m_1, m_2):
    model_1, count_1 = m_1
    model_2, count_2 = m_2
    model_1.coef_ += model_2.coef_
    model_1.intercept_ += model_2.intercept_
    return model_1, count_1 + count_2


def parallel_train(model, data, classes=None, n_iter=10):
    for i in range(n_iter):
        models = data.mapPartitions(lambda x: _train(x, model, classes))
        model, count = models.reduce(_model_sum)
        model.coef_ /= count
        model.intercept_ /= count
    return model


def _train_cv(iterator, models, classes):
    for i, (X, y) in enumerate(iterator):
        for j, model in enumerate(models):
            if i % len(models) != j:
                model.partial_fit(X, y, classes=classes)
    yield models, 1


def _predict_cv(iterator, models):
    for i, (X, y) in enumerate(iterator):
        predictions = []
        for j, model in enumerate(models):
            if i % len(models) == j:
                y_predicted = model.predict(X)
                predictions.append((y, y_predicted))
        yield tuple(predictions)


def _models_sum(m_1, m_2):
    models_1, count_1 = m_1
    models_2, count_2 = m_2
    for model_1, model_2 in zip(models_1, models_2):
        model_1.coef_ += model_2.coef_
        model_1.intercept_ += model_2.intercept_
    return models_1, count_1 + count_2


def _accuracy_sum(p_1, p_2):
    pred_1, count_1 = p_1
    pred_2, count_2 = p_2
    diffs = [(true != pred)]


def parallel_cross_validation(model, data, classes=None, n_folds=5, n_iter=10):
    # train one model per fold
    models = [clone(model) for i in range(n_folds)]
    for i in range(n_iter):
        train = lambda x: _train_cv(x, models, classes)
        models, count = data.mapPartitions(train).reduce(_models_sum)
        for model in models:
            model.coef_ /= count
            model.intercept_ /= count

    # compute predictions for each model
    predict = lambda x: _predict_cv(x, models)
    accuracies = data.mapPartitions(predict).reduce(_accuracy_sum)