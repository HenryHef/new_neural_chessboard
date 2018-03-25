


from scipy.optimize._minimize import minimize

import numpy as np


def crossproduct(v1,v2):
	return (v1[0]*v2[1]) - (v1[1]*v2[0])

def fit_affine(X,Y):
	Y=np.array([[y[0],y[1],1.0] for y in Y]).transpose()
	X=np.array([[x[0],x[1],1.0] for x in X]).transpose()
	return Y.dot(np.linalg.inv(X))

def fit_perspective_transform(X,A,guess):
	X_=np.zeros((len(X),3))
	for i in range(len(X)):
		X_[i][0]=X[i][0]
		X_[i][1]=X[i][1]
		X_[i][2]=1.0
	X=np.array(X_)
	A=np.array(A)
	def y(M):
		re=0
		for n in range(X.shape[0]):
			u=M.dot(X[n])
			u=[u[0]/u[2],u[1]/u[2]]
			re+=(u[0]-A[n][0])**2+(u[1]-A[n][1])**2
		return re
	def y_multivar(i):
		return y(np.array([[i[0],i[1],i[2]],[i[3],i[4],i[5]],[i[6],i[7],i[8]]]))
	def p_y_p_Mij(M,i,j):
		delta=.001
		delMij=np.zeros(M.shape)
		delMij[i,j]=delta
		return (y(M+delMij)-y(M))/delta
	
	def grad_y_by_M(M):
		re=np.zeros(M.shape)
		for i in range(M.shape[0]):
			for j in range(M.shape[1]):
				re[i,j]=p_y_p_Mij(M,i,j)
		return re
	
	
	M=minimize(y_multivar, [guess[0,0],guess[0,1],guess[0,2]
						,guess[1,0],guess[1,1],guess[1,2],
						0,0,1], args=())['x']
	M=np.array([[M[0],M[1],M[2]],[M[3],M[4],M[5]],[M[6],M[7],M[8]]])
	M/=M[2,2]

	return M
def find_3_non_colin(pts):
	#index 0
	re=[]
	for i2 in range(1,len(pts)):
		for i3 in range(i2+1,len(pts)):
			v2=(1.0*pts[i2][0]-pts[0][0],1.0*pts[i2][1]-pts[0][1])
			v3=(1.0*pts[i3][0]-pts[0][0],1.0*pts[i3][1]-pts[0][1])
			re.append((i2,i3,crossproduct(v2,v3)))
	re.sort(key=lambda x:abs(x[2]))
	return [0,re[-1][0],re[-1][1]]

def proj(M,v):
	v=M.dot(np.array([v[0],v[1],1.0]))
	return (v[0]/v[2],v[1]/v[2])
def inv_proj(M,v):
	v=np.linalg.inv(M).dot(np.array([v[0],v[1],1.0]))
	return (v[0]/v[2],v[1]/v[2])

def get_transform(from_pts,to_pts):
	colin_idxs = find_3_non_colin(to_pts)
	
	three_from = [from_pts[colin_idx] for colin_idx in colin_idxs]
	three_to = [from_pts[colin_idx] for colin_idx in colin_idxs]
	
	affine_guess = fit_affine(three_from,three_to)
	M=fit_perspective_transform(from_pts,to_pts,affine_guess)
	return M

						
