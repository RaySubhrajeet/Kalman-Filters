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
		predict_mean=[]
		predict_cov=[]
		for z in sensor_data:
			self._predict()     
			self._update(z)     
			predict_mean.append(self.mu)        
			predict_cov.append(self.sigma)
		
		return (predict_mean,predict_cov)

	

	def _predict(self):
		# FILL in your code here
		self.mu=self.g(self.mu)
		self.sigma = np.dot(self.g_jac(self.mu), np.dot(self.sigma, self.g_jac(self.mu).T)) + self.R

	def _update(self, z):
		# FILL in your code here
		S = self.Q + np.dot(self.h_jac(self.mu), np.dot(self.sigma, self.h_jac(self.mu).T))

		K_gain = np.dot(np.dot(self.sigma, self.h_jac(self.mu).T), 1/S)
	   
		y = z - self.h(self.mu)
	   
	   
		I = np.eye(2)
		self.sigma = np.dot(I - np.dot(K_gain, self.h_jac(self.mu)), self.sigma)
		self.mu = self.mu + np.dot(K_gain, y)


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


def g(x):
	return np.transpose(np.array([x[0]* x[1], x[1]]))

def h(x):
	return np.sqrt(x[0]**2 + 1)

def h_jac(x):
	""" compute Jacobian of H matrix at x """
		
	denom = np.sqrt(x[0]**2 + 1)
	return np.array ([x[0]/denom, 0])

def g_jac(x):
	 return np.array([[x[1],0 ], 
					[0, 1]])

def problem3():
	# FILL in your code here
	
	ground_truth=[]
	t = np.linspace(1, 20, 20)
	for time in t:
		ground_truth.append(h(t))
		ground_truth.append(0.1) 
	measurement = h(t) + np.random.normal(0, 1.0, 20)

	sigma=np.array([[2, 0],
					[0, 2]])
	mu=np.array([1,2])
	mu=mu.T
	 
	Q = 1.0
	R=np.array([[0.25, 0],
				[0, 0.25]])
	
	ekf = ExtendedKalmanFilter(mu = mu, sigma = sigma,g= g ,g_jac = g_jac,h =h,h_jac= h_jac,R= R, Q=Q) 
	extended_kalman_output=ekf.run(measurement)
	
	
	predict_mean=np.asarray(extended_kalman_output[0])
	predict_mean=predict_mean.reshape(20,2)

	predict_cov=np.asarray(extended_kalman_output[1])
	predict_cov=predict_cov.reshape(20,2,2)

	ground_truth=np.asarray(ground_truth)
	ground_truth=ground_truth.reshape(20,2)

	plot_prediction(t,ground_truth,predict_mean,predict_cov)

if __name__ == '__main__':
	problem3()
