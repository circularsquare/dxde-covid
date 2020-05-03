from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
import pickle

def gen_train_data():
    data = np.load('sir.dat.npy')
    return data[:, 0:1], data[:, 1:4]
def data_sum(path):
    df = pd.read_csv(path)
    df = df[df['Province/State']=='Hubei'].transpose().iloc[4:, :]
    df['sum'] = df.sum(axis=1)
    return df.iloc[:, -1].to_numpy()

def get_covid_data():
    path1 = './data/time_series_covid19_confirmed_global.csv'
    path2 = './data/time_series_covid19_recovered_global.csv'
    path3 = './data/time_series_covid19_deaths_global.csv'
    inf = data_sum(path1)
    rem = np.add(data_sum(path2), data_sum(path3))
    sus = np.array([52e6-rem[i] for i in range(len(inf))])
    ind = np.array([[i+0.0] for i in range(len(inf))])
    inf2 = np.array([inf[i]-rem[i] for i in range(len(inf))])
    fig, ax = plt.subplots()
    ax.set_yscale('symlog')
    plt.plot(rem)
    plt.plot(sus)
    plt.plot(inf2)
    return ind.astype('float32'), np.stack([sus, inf2, rem], axis=1).astype('float32')

def main():
    data = get_covid_data()
    #sir params
    beta = tf.Variable(.0001)
    gamma = tf.Variable(.0002)
    N = data[1][0][0]+data[1][0][1] #population
    initial_infected = data[1][0][1]
    initial_removed = data[1][0][2]

    #x is time, y is odes
    def odes(x, y):
        s, i, r = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        ds_x = tf.gradients(s, x)[0]
        di_x = tf.gradients(i, x)[0]
        dr_x = tf.gradients(r, x)[0]
        de1 = ds_x + abs(beta)*i*s/N
        de2 = di_x - abs(beta)*i*s/N + (abs(gamma)*i)
        de3 = dr_x - abs(gamma)*i
        return [de1, de2, de3]

    def boundary(_, on_initial):
        return on_initial

    def func(x):
        return np.hstack((np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape)))

    geom = dde.geometry.TimeDomain(0, len(data[0]))
    ic1 = dde.IC(geom, lambda x: (N-initial_infected-initial_removed) * np.ones(x.shape), boundary, component=0)
    ic2 = dde.IC(geom, lambda x: initial_infected * np.ones(x.shape), boundary, component=1)
    ic3 = dde.IC(geom, lambda x: initial_removed * np.ones(x.shape), boundary, component=2)

    observe_t, observe_y = data
    ptset = dde.bc.PointSet(observe_t)
    inside = lambda x, _: ptset.inside(x)
    observe_y0 = dde.DirichletBC(
        geom, ptset.values_to_func(observe_y[:, 0:1]), inside, component=0
    )
    observe_y1 = dde.DirichletBC(
        geom, ptset.values_to_func(observe_y[:, 1:2]), inside, component=1
    )
    observe_y2 = dde.DirichletBC(
        geom, ptset.values_to_func(observe_y[:, 2:3]), inside, component=2
    )

    data = dde.data.PDE(geom, odes, [ ic2, ic3, observe_y1, observe_y2], num_domain=300, num_boundary=1, anchors=observe_t)
    net = dde.maps.FNN([1] + [40] * 4 + [3], 'tanh', 'Glorot uniform')
    model = dde.Model(data, net)

    model.compile("adam", lr=.01, loss='mse') # try 'log cosh'
    variable = dde.callbacks.VariableValue(
        [beta, gamma], period=1000, filename = "variables.dat"
    )
    losshistory, train_state = model.train(epochs=10000, callbacks=[variable])
    plot_best_state(train_state)
    plot_loss_history(losshistory)
    plt.show()




def plot_best_state(train_state):
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()

    y_dim = best_y.shape[1]

    # Regression plot
    if X_test.shape[1] == 1:
        idx = np.argsort(X_test[:, 0])
        X = X_test[idx, 0]
        plt.Figure()
        for i in range(y_dim):
            if y_train is not None:
                plt.plot(X_train[:, 0], y_train[:, i], 'o', color=(.6, .4+i*.3, .4+i*.3), label="Train" + str(i))
            if y_test is not None:
                plt.plot(X, y_test[idx, i], '-', color=(.6, .4+i*.3, .4+i*.3), label="True" + str(i))
            plt.plot(X, best_y[idx, i], '-', color=(.2, i*.4, i*.4), label="Prediction" + str(i))
            if best_ystd is not None:
                plt.plot(
                    X, best_y[idx, i] + 2 * best_ystd[idx, i], "-b", label="95% CI"
                )
                plt.plot(X, best_y[idx, i] - 2 * best_ystd[idx, i], "-b")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
    elif X_test.shape[1] == 2:
        for i in range(y_dim):
            plt.figure()
            ax = plt.axes(projection=Axes3D.name)
            ax.plot3D(X_test[:, 0], X_test[:, 1], best_y[:, i], ".")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$y_{}$".format(i + 1))

    # Residual plot
    if y_test is not None:
        plt.figure()
        residual = y_test[:, 0] - best_y[:, 0]
        plt.plot(best_y[:, 0], residual, "o", zorder=1)
        plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles="dashed", zorder=2)
        plt.xlabel("Predicted")
        plt.ylabel("Residual = Observed - Predicted")
        plt.tight_layout()

    if best_ystd is not None:
        plt.figure()
        for i in range(y_dim):
            plt.plot(X_test[:, 0], best_ystd[:, i], "-b")
            plt.plot(
                X_train[:, 0],
                np.interp(X_train[:, 0], X_test[:, 0], best_ystd[:, i]),
                "ok",
            )
        plt.xlabel("x")
        plt.ylabel("std(y)")
def plot_loss_history(losshistory):
    loss_train = np.sum(
        np.array(losshistory.loss_train) * losshistory.loss_weights, axis=1
    )
    loss_test = np.sum(
        np.array(losshistory.loss_test) * losshistory.loss_weights, axis=1
    )

    plt.figure()
    plt.semilogy(losshistory.steps, loss_train, label="Train loss")
    plt.semilogy(losshistory.steps, loss_test, label="Test loss")
    for i in range(len(losshistory.metrics_test[0])):
        plt.semilogy(
            losshistory.steps,
            np.array(losshistory.metrics_test)[:, i],
            label="Test metric",
        )
    plt.xlabel("# Steps")
    plt.legend()

if __name__ == "__main__":
    main()
    # get_covid_data()
    # gen_traindata()
    pass
