
def organize_data(test_score_matriz, Y_test, train_score_matriz, Y_train):
    """
    Return organize dictionary by ["test"|"train"] =>{"data":, "label":}
    :param test_score_matriz:
    :param Y_test:
    :param train_score_matriz:
    :param Y_train:
    :return:
    """
    data = {}
    data["test"] = {}
    data["train"] = {}
    data["test"]["data"] = test_score_matriz
    data["test"]["label"] = Y_test
    data["train"]["data"] = train_score_matriz
    data["train"]["label"] = Y_train

    return data
