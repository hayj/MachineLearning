import numpy as np
from operator import itemgetter

def softmax_mean_rank(y_true, y_pred):
    """
        This function calcute the mean rank of the true category
        A score near to 1 means the model always predict a good softmax
        Near to 0.5 --> random softmax...
    """
    scores = []
    for i in range(len(y_true)):
        t = y_true[i]
        p = y_pred[i]
        position = np.where(t == 1)[0][0]
        p = [(x, u) for u, x in enumerate(p)]
        p = sorted(p, key=itemgetter(0), reverse=True)
        rank = 0
        u = 0
        for current in p:
            if current[1] == position:
                rank = u
                break
            u += 1
        score = rank / (len(p) - 1)
        scores.append(score)
    return 1.0 - np.mean(scores)


if __name__ == '__main__':
    print(softmax_mean_rank\
    (
        np.array(
        [
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
        ]),
        np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.6, 0.8, 0.1],
            [0.1, 0.3, 0.1, 0.1, 0.6, 0.4, 0.1],
            [0.1, 0.5, 0.0, 0.1, 0.6, 0.4, 0.9],
        ])
    ))
    print(softmax_mean_rank\
    (
        np.array(
        [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
        ]),
        np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.6, 0.8, 0.1],
            [0.1, 0.3, 0.1, 0.1, 0.6, 0.4, 0.1],
            [0.1, 0.5, 0.0, 0.1, 0.6, 0.4, 0.9],
        ])
    ))
    print(softmax_mean_rank\
    (
        np.array(
        [
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
        ]),
        np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.6, 0.8, 0.1],
            [0.1, 0.3, 0.1, 0.1, 0.6, 0.4, 0.1],
            [0.1, 0.5, 0.0, 0.1, 0.6, 0.4, 0.9],
        ])
    ))