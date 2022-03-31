from sklearn.utils import Bunch


def transform(samples, y):
    x_transformed = samples.to_numpy()
    y_transformed = y.to_numpy()
    return Bunch(data=x_transformed, target=y_transformed)


def transform_original_complete(original, original_y, complete, complete_y):
    return transform(original, original_y), transform(complete, complete_y)


def delete_multiple_element(list_object, indices, reverse=True):
    indices = sorted(indices, reverse=reverse)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
