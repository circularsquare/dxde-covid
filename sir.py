#solves SIR system given coefficients'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde

def main():
    #sir params
    beta = 1 #infection
    gamma = .2 #removal
    N = 150 #population
    initial_infected = 20


    #x is time, y is odes
    def odes(x, y):
        s, i, r = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        ds_x = tf.gradients(s, x)[0]
        di_x = tf.gradients(i, x)[0]
        dr_x = tf.gradients(r, x)[0]
        de1 = ds_x + beta*i*s/N
        de2 = di_x - (beta*i*s/N) + (gamma*i)
        de3 = dr_x - gamma*i
        # de1 = tf.gradients(ds_x, x)[0] - s
        # de2 = tf.gradients(di_x, x)[0] - i
        # de3 = tf.gradients(dr_x, x)[0] - r
        return [de1, de2, de3]

    def boundary(_, on_initial):
        return on_initial

    def func(x):
        return np.hstack((0*np.cos(x), 0*np.cos(x), 0*np.cos(x)))
        #return np.hstack(((N-initial_infected)*np.cos(x), initial_infected*np.cos(x), 0*np.cos(x)))

    geom = dde.geometry.TimeDomain(0, 15)
    ic1 = dde.IC(geom, lambda x: (N-initial_infected) * np.ones(x.shape), boundary, component=0)
    ic2 = dde.IC(geom, lambda x: initial_infected * np.ones(x.shape), boundary, component=1)
    ic3 = dde.IC(geom, lambda x: 0 * np.ones(x.shape), boundary, component=2)
    data = dde.data.PDE(geom, odes, [ic1, ic2, ic3], 1000, 3)

    layer_size = [1] + [50] * 3 + [3]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)

    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(epochs=20000)

    #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
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
        XandY = np.concatenate((X_test, best_y), axis=1)
        np.save('sir.dat', XandY)


    elif X_test.shape[1] == 2:
        for i in range(y_dim):
            plt.figure()
            ax = plt.axes(projection=Axes3D.name)
            ax.plot3D(X_test[:, 0], X_test[:, 1], best_y[:, i], ".")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$y_{}$".format(i + 1))

    # Residual plot
    '''if y_test is not None:
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
        plt.ylabel("std(y)")'''
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
