import matplotlib.pyplot as plt
import numpy as np


class ExtendedKalmanFilter():
    """
    Implementation of an Extended Kalman Filter.
    """
    def __init__(self, mu, sigma, g, g_jac, h, h_jac, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param g: process function
        :param g_jac: process function's jacobian
        :param h: measurement function
        :param h_jac: measurement function's jacobian
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.g = g
        self.g_jac = g_jac
        self.R = R
        # measurement model
        self.h = h
        self.h_jac = h_jac
        self.Q = Q

    def reset(self):
        """
        Reset belief state to initial value.
        """
        self.mu = self.mu_init
        self.sigma = self.sigma_init

    def run(self, sensor_data):
        """
        Run the Kalman Filter using the given sensor updates.

        :param sensor_data: array of T sensor updates as a TxS array.

        :returns: A tuple of predicted means (as a TxD array) and predicted
                  covariances (as a TxDxD array) representing the KF's belief
                  state AFTER each update/predict cycle, over T timesteps.
        """
        # FILL in your code here

    def _predict(self):
        # FILL in your code here

    def _update(self, z):
        # FILL in your code here


def plot_prediction(t, ground_truth, predict_mean, predict_cov):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    gt_x, gt_a = ground_truth[:, 0], ground_truth[:, 1]
    pred_x, pred_a = predict_mean[:, 0], predict_mean[:, 1]
    pred_x_std = np.sqrt(predict_cov[:, 0, 0])
    pred_a_std = np.sqrt(predict_cov[:, 1, 1])

    plt.figure(figsize=(7, 10))
    plt.subplot(211)
    plt.plot(t, gt_x, color='k')
    plt.plot(t, pred_x, color='g')
    plt.fill_between(
        t,
        pred_x-pred_x_std,
        pred_x+pred_x_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$x$")
    plt.title(r"EKF estimation: $x$")

    plt.subplot(212)
    plt.plot(t, gt_a, color='k')
    plt.plot(t, pred_a, color='g')
    plt.fill_between(
        t,
        pred_a-pred_a_std,
        pred_a+pred_a_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$\alpha$")
    plt.title(r"EKF estimation: $\alpha$")

    plt.show()


def problem3():
    # FILL in your code here


if __name__ == '__main__':
    problem3()
