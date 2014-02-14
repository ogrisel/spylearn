"""Parallel Linear Model training with partial_fit and averaging"""


def _train(iterator, model, classes):
    for X, y in iterator:
        model.partial_fit(X, y, classes=classes)
    yield model, 1


def _model_sum(m_1, m_2):
    model_1, count_1 = m_1
    model_2, count_2 = m_2
    model_1.coef_ += model_2.coef_
    model_1.intercept_ += model_2.intercept_
    return model_1, count_1 + count_2


def parallel_train(model, data, classes):
    models = data.mapPartitions(lambda x: _train(x, model))
    model, count = models.reduce(_model_sum)
    model.coef_ /= count
    model.intercept_ /= count
    return model
