import numpy as np

"""
Method Name : gaussian_filter()
input parameters: n - size of gaussian kernel
                  sigma - Gaussian sigma for gaussian kernel
output variables : kernel - Gaussian kernel with given size and sigma, 
Note: The values in the kernel obtained here does not sum up to one.
"""
def gaussian_filter(n,sigma):
	k = floor(n/2)
	kernel = np.zeros((n,n))
	for i in range(0,n):
		for j in range(0,n):
			x = i-k
			y = j-k
			kernel[i,j] =  e** -((x**2+y**2)/(2*(sigma**2)))
	kernel = np.round(kernel/kernel[0,0])
	return kernel

"""
Method Name : gradient
input parameters: im - image.
                  operator_type - String, by default it will use prewitt filter, 
	                            if one wants to use Sobel filter they have to 
	                            mention while calling like operator_type = 'sobel'
output variables : g_mag - gradient magnitude
                   g_dir - gradient direction
"""
def gradient(im,operator_type='prewitt'):
	# calling gradient_operator for generating horizontal and vertical gradient operators
	op_x,op_y = gradient_operator(operator_type)
	# cross correlation of image with horizontal gradient operator
	g_x = cross_correlation(im,op_x)
	#cross correlation of image with Vertical gradient operator
	g_y = cross_correlation(im,op_y)
	#gradient magnitude
	g_mag = np.sqrt(np.square(g_x)+np.square(g_y))
	#gradient direction, g_dir will have values in radians and in the range of (-180 degrees to 180 degrees but in radians)  
	g_dir = np.arctan2(g_y,g_x)
	return g_x,g_y,g_mag,g_dir

"""
Method Name : gradient_operator() - generates horizontal and vertical gradient operators
input parameters: operator_type - it's a string, expected values are 'prewitt' or 'sobel'
output variables : op_x - horizontal gradient operator
                   op_y - vertical gradient operator
"""
def gradient_operator(operator_type):
	p = np.ones((1,3))
	q = np.array([1,0,-1])[np.newaxis,:]
	if(operator_type.lower() == "prewitt"):
		op_x = -1*p.T @ q
		op_y = q.T @ p
	if(operator_type.lower() == "sobel"):
		p[0,1] =2
		op_x = -1*p.T @ q
		op_y = q.T @ p
	return op_x,op_y