import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.metrics import Recall
from tensorflow.keras import regularizers, saving, callbacks
import math

history_seconds = int(sys.argv[1])
next_prediction_period = int(sys.argv[2])
input_shape = history_seconds // next_prediction_period - 1


EPSILON = 1e-6

dir = os.path.dirname(__file__)


def get_x_y(data):
    seq = []
    y = []
    last_wl = []
    for i in range(0, len(data) - history_seconds, next_prediction_period):
        t = data[i:i + history_seconds + next_prediction_period]
        temp_x = []
        for j in range(0, len(t), next_prediction_period):
            temp_x.append(max(t[j:j + next_prediction_period]))
        x = []
        for j in range(1, len(temp_x)):
            x.append(np.log(temp_x[j] + EPSILON) - np.log(temp_x[j-1] + EPSILON))

        seq.append(x[:-1])
        y.append(x[-1])
        last_wl.append(temp_x[-2])

    
    return tf.convert_to_tensor(seq, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32), last_wl


def get_data():
    with open(f'{os.path.dirname(__file__)}/workload.txt', "r") as f:
        workload = f.readlines()
    workload = workload[0].split()
    workload = list(map(int, workload))

    workload = list(filter(lambda x: x != 0, workload))
    train_to_idx = 14 * 24 * 60 * 60
    workload_train = workload[:train_to_idx]
    workload_test = workload[train_to_idx:]

    train_x, train_y, _ = get_x_y(workload_train)
    test_x, test_y, _ = get_x_y(workload_test)
    return (
        train_x,
        train_y,
        test_x,
        test_y
    )


def create_model():
    sequence_input = Input(shape=(input_shape, 1), name="sequence_input")
    x = LSTM(25, activation="relu", kernel_regularizer=regularizers.L1(0.00001))(sequence_input)
    output = Dense(1)(x)
    model = Model(inputs=sequence_input, outputs=output)
    return model


def my_train():
    tf.random.set_seed(7)
    train_x, train_y, test_x, test_y = get_data()
    print("shapes")
    print(len(train_x),)
    print(train_y.shape)
    print(len(test_x),)
    print(test_y.shape)
    model = create_model()
    print("model summary")
    print(model.summary())
    print("now compiling")
    model.compile(optimizer="adam", loss="mse")
    print("callback definition")
    callback = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
    print("now fitting")
    model.fit(train_x, train_y, epochs=50, batch_size=16, validation_data=(test_x, test_y), callbacks=[callback])
    
    model.save(f'{dir}/lstm_changes_rate_h{history_seconds}_s{next_prediction_period}.keras')


def my_test():
    model = saving.load_model(f'{dir}/lstm_changes_rate_h{history_seconds}_s{next_prediction_period}.keras')
    with open(f"{os.path.dirname(__file__)}/workload.txt", "r") as f:
        wl = f.read()
    
    wl = list(map(lambda x: round(int(x)), wl.split()))
    wl = list(filter(lambda x: x != 0, wl))

    hour = 60 * 60
    day = hour * 24
     
    # test_data = wl[15 * day + 84 * 60 + 450: 15 * day + 95 * 60 + 450]
    test_data = wl[15 * day: 16*day]
    test_x, test_y, last_wl = get_x_y(test_data)

    predictions = model.predict(test_x)
    print(len(test_data), len(test_x), len(test_y), len(predictions))

    predicted_wl = []
    actual_wl = []
    for i in range(len(test_y)):
        predicted_wl.append(last_wl[i] * math.e ** (predictions[i]))
        actual_wl.append(last_wl[i] * math.e ** (test_y[i]))
    
    plt.plot(list(range(len(test_y))), predicted_wl, label="predicted")
    plt.plot(list(range(len(test_y))), actual_wl, label="actual")
    plt.legend()
    plt.savefig(f"{dir}/lstm_changes_rate_{history_seconds}_s{next_prediction_period}.pdf", dpi=600, format="pdf", pad_inches=0, bbox_inches="tight")

    errors = [abs(y_t - y_p) for y_t, y_p in zip(actual_wl, predicted_wl)]
    squared_errors = [e**2 for e in errors]
    mae = sum(errors) / len(squared_errors)
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
    print(f"MAE: {mae}, RMSE: {rmse}")
    
    print("total days of workload", len(wl) / day)




if __name__ == "__main__":
    # my_train()
    my_test()

