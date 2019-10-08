import matplotlib.pyplot as plt
import numpy as np
from numpy import dot 


class KalmanFilter():
	"""
	Implementation of a Kalman Filter.
	"""
	def __init__(self, mu, sigma, A, C, R=0., Q=0.):
		"""
		:param mu: prior mean
		:param sigma: prior covariance
		:param A: process model
		:param C: measurement model
		:param R: process noise
		:param Q: measurement noise
		"""
		# prior
		self.mu = mu
		self.sigma = sigma
		self.mu_init = mu
		self.sigma_init = sigma
		# process model
		self.A = A
		self.R = R
		# measurement model
		self.C = C
		self.Q = Q

		#dimensions
		self.n = A.shape[1]
		self.m = C.shape[1]

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
		  self.mu= dot(self.A,self.mu)
		  self.sigma = dot(self.A, dot(self.sigma, self.A.T)) + self.R
		  

	def _update(self, z):
		# FILL in your code here
	   
		S = self.Q + np.dot(self.C, np.dot(self.sigma, self.C.T))
		K_gain = np.dot(np.dot(self.sigma, self.C.T), np.linalg.inv(S))
	   
		y = z - np.dot(self.C, self.mu)
	   
		self.mu = self.mu + np.dot(K_gain, y)
		I = np.eye(self.n)
		self.sigma = np.dot(I - np.dot(K_gain, self.C), self.sigma)
			



def plot_prediction(t, ground_truth, measurement, predict_mean, predict_cov):
	"""
	Plot ground truth vs. predicted value.

	:param t: 1-dimensional array representing timesteps, in seconds.
	:param ground_truth: Tx1 array of ground truth values
	:param measurement: Tx1 array of sensor values
	:param predict_mean: TxD array of mean vectors
	:param predict_cov: TxDxD array of covariance matrices
	"""


	predict_pos_mean = predict_mean[:, 0]
	predict_pos_std = predict_cov[:, 0, 0]

	

	plt.figure()
	plt.plot(t, ground_truth, color='k')
	plt.plot(t, measurement, color='r')
	plt.plot(t, predict_pos_mean, color='g')
	plt.fill_between(
		t,
		predict_pos_mean-predict_pos_std,
		predict_pos_mean+predict_pos_std,
		color='g',
		alpha=0.5)
	plt.legend(("ground truth", "measurements", "predictions"))
	plt.xlabel("time (s)")
	plt.ylabel("position (m)")
	plt.title("Predicted Values")
	plt.show()


def plot_mse(t, ground_truth, predict_means):
	"""
	Plot MSE of your KF over many trials.

	:param t: 1-dimensional array representing timesteps, in seconds.
	:param ground_truth: Tx1 array of ground truth values
	:param predict_means: NxTxD array of T mean vectors over N trials
	"""
	predict_pos_means = predict_means[:, :, 0]
	errors = ground_truth.squeeze() - predict_pos_means
	mse = np.mean(errors, axis=0) ** 2

	plt.figure()
	plt.plot(t, mse)
	plt.xlabel("time (s)")
	plt.ylabel("position MSE (m^2)")
	plt.title("Prediction Mean-Squared Error")
	plt.show()


def problem2a():
	# FILL in your code here

	A = np.array([[1, 0.1, 0, 0], 
				  [0, 1, 0.1, 0],
				  [0, 0, 1, 0.1],
				  [0, 0, 0, 1]])

	C = np.array([1, 0, 0, 0]).reshape(1, 4)
	mu=np.transpose(np.array([5,1,0,0]))

	sigma=np.array([[10, 0, 0, 0],
					[0, 10, 0, 0],
					[0, 0, 10, 0],
					[0, 0, 0, 10]])
	Q = 1.0
		
	t = np.linspace(0.1, 10, 100)
	ground_truth=np.sin(t) 
	measurement = np.sin(t) + np.random.normal(0, 1.0, 100)
	predict_mean=[]
	predict_cov=[]	

	kf = KalmanFilter(A = A, C = C, mu = mu, sigma = sigma,Q=Q)	
	kalman_output=kf.run(measurement)
	
	predict_mean=np.asarray(kalman_output[0])	
	predict_mean=predict_mean.reshape(100,4)

	predict_cov=np.asarray(kalman_output[1])
	predict_cov=predict_cov.reshape(100,4,4)
	
	
	plot_prediction(t, ground_truth, measurement, predict_mean, predict_cov)


	#Calculating Mean Squared Error    
	kf.reset()
	predict_mean=[]
	predict_cov=[]
	for x in range(0, 10000):	

		kalman_output=kf.run(measurement)
		predict_mean.append(kalman_output[0])		
		predict_cov.append(kalman_output[1])
		kf.reset()
		
	predict_mean=np.asarray(predict_mean)
	predict_mean=predict_mean.reshape(10000,100,4)
	plot_mse(t,ground_truth,predict_mean)	
	

def problem2b():
	# FILL in your code here
	A = np.array([[1, 0.1, 0, 0], 
				  [0, 1, 0.1, 0],
				  [0, 0, 1, 0.1],
				  [0, 0, 0, 1]])

	C = np.array([1, 0, 0, 0]).reshape(1, 4)
	mu=np.transpose(np.array([5,1,0,0]))

	sigma=np.array([[10, 0, 0, 0],
					[0, 10, 0, 0],
					[0, 0, 10, 0],
					[0, 0, 0, 10]])
	Q = 1.0
	R=np.array([[0.1, 0, 0, 0],
				[0, 0.1, 0, 0],
				[0, 0, 0.1, 0],
				[0, 0, 0, 0.1]])
	#R = np.array([0.5]).reshape(1, 1)
	
	N=10000
	t = np.linspace(0.1, 10, 100)
	ground_truth=np.sin(t) 
	measurement = np.sin(t) + np.random.normal(0, 1.0, 100)
	predict_mean=[]
	predict_cov=[]	

	kf = KalmanFilter(A = A, C = C, mu = mu, sigma = sigma,R=R, Q=Q)
	for x in range(0, 10000):	

		kalman_output=kf.run(measurement)
		predict_mean.append(kalman_output[0])		
		predict_cov.append(kalman_output[1])
		kf.reset()
		
	predict_mean=np.asarray(predict_mean)
	predict_mean=predict_mean.reshape(10000,100,4)

	
	plot_mse(t,ground_truth,predict_mean)	


if __name__ == '__main__':
	problem2a()
	problem2b()
