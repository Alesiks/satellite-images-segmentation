from keras.layers import K

smooth = 1e-12

def jaccard_coef(y_true_values, y_predictions):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true_values * y_predictions, axis=[0, -1, -2])
    sum_ = K.sum(y_true_values + y_predictions, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true_values, y_predictions):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_predictions, 0, 1))

    intersection = K.sum(y_true_values * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true_values + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)