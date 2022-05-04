from sklearn.utils import Bunch


def transform(samples, y):
    """
    Transform pandas DataFrame to Sklearn Bunch.

    It takes a pandas dataframe and a pandas series, converts them to numpy
    arrays, and returns a Bunch object with the data and target attributes
    set to the numpy arrays

    :param samples: The data to be transformed
    :param y: The target variable
    :return: A Bunch object, which is a dictionary-like object.
    """
    x_transformed = samples.to_numpy()
    y_transformed = y.to_numpy()
    return Bunch(data=x_transformed, target=y_transformed)


def transform_original_complete(original, original_y, complete, complete_y):
    """
    It takes in two sets of data, and returns two sets of data.

    :param original: the original data
    :param original_y: the original labels
    :param complete: the complete data set
    :param complete_y: the labels for the complete data
    :return: The transformed original and complete data.
    """
    return transform(original, original_y), transform(complete, complete_y)


def delete_multiple_element(list_object, indices, reverse=True):
    """
    It deletes multiple elements from a list, given a list of indices.

    :param list_object: The list object you want to delete elements from
    :param indices: a list of indices to be deleted
    :param reverse: If True, the indices are sorted in descending order. If
    False, the indices are sorted in ascending order, defaults to True (
    optional)
    """
    indices = sorted(indices, reverse=reverse)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
