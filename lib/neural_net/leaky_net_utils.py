import tensorflow as tf

from lib.neural_net.leaky_relu_decision_net import DecisionNeuralNet as \
    DecisionNeuralNet_leaky_relu_3layers_with_sigmoid
from lib.utils.evaluation_utils import simple_evaluation_output


def train_leaky_neural_net_various_tries_over_svm_output(
        decision_net_session_conf,
        architecture, HYPERPARAMS_decision_net, train_score_matriz,
        test_score_matriz,
        Y_train, Y_test, bool_test=False):
    temp_results_per_try_test = []

    temp_results_per_try_train = []
    for i in range(1, decision_net_session_conf['decision_net_tries'] + 1, 1):

        print("Neural net try: {}".format(i))
        tf.reset_default_graph()
        v = DecisionNeuralNet_leaky_relu_3layers_with_sigmoid(
            architecture=architecture,
            hyperparams=HYPERPARAMS_decision_net,
            bool_test=bool_test)

        v.train(train_score_matriz, Y_train,
                max_iter=decision_net_session_conf['max_iter'],
                iter_to_show_error=10)
        print("Net Trained")

        # Test net created
        score_train = v.forward_propagation(train_score_matriz)[0]
        score_test = v.forward_propagation(test_score_matriz)[0]

        # Fixing the threeshold in function of the test evaluation or
        # prefixed it to 0.5, the last door of the neural net is a sigmoid

        if decision_net_session_conf["threshould_prefixed_to_0.5"]:

            threshold, decision_net_dic_train = simple_evaluation_output(
                score_train,
                Y_train, thresholds_establised=0.5,
                bool_test=bool_test)
        else:

            threshold, decision_net_dic_train = simple_evaluation_output(
                score_train,
                Y_train, bool_test=bool_test)

        _, decision_net_dic_test = simple_evaluation_output(
            score_test, Y_test, thresholds_establised=0.5,
            bool_test=bool_test)

        temp_results_per_try_test.append(decision_net_dic_test)
        temp_results_per_try_train.append(decision_net_dic_train)

    try_selected_test_dic = sorted(temp_results_per_try_test,
                                   key=lambda results: results[
                                       decision_net_session_conf[
                                           'field_to_select_try']],
                                   reverse=True)[0]

    try_selected_train_dic = sorted(temp_results_per_try_train,
                                    key=lambda results: results[
                                        decision_net_session_conf[
                                            'field_to_select_try']],
                                    reverse=True)[0]

    if bool_test:
        print("Results over 10 tries over decision neural net:")
        for i in range(0, len(temp_results_per_try_test), 1):
            print("Test: " + str(temp_results_per_try_test[i]))
            print("Train: " + str(temp_results_per_try_train[i]))

    print("Intent selected:")
    print("Decision Neural Net Test: " + str(try_selected_test_dic))
    print("Decision Neural Net Train: " + str(try_selected_train_dic))

    return try_selected_train_dic, try_selected_test_dic
