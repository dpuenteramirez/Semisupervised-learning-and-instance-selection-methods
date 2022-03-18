from sklearn.utils import Bunch


def transform(X, y):
    X_transformed = X.to_numpy()
    y_transformed = y.to_numpy()
    return Bunch(data=X_transformed, target=y_transformed)


def delete_multiple_element(list_object, indices, reverse=True):
    indices = sorted(indices, reverse=reverse)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
