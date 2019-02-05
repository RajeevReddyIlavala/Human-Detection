from PIL import Image
from pylab import *
from kernels import *


"""
Method Name : cross_correlation() - performs cross-correlation operation on the image with the given kernel
                                    Here it is used for calculating horizontal and vertical 
                                    gradient responses.
Input parameters : im - input image
                   kernel - can be Horizontal/Vertical gradient operator
"""
def cross_correlation(im,kernel):
	im_correlated = im.copy()
	# if kernel size is 7*7 then k = 3, this is to consider the neighbours of center pixel
	k = int(floor(kernel.shape[0]/2))
	# The below for loop performs the cross-correlation.
	for i in range(k,im.shape[0]-k):
		for j in range(k,im.shape[1]-k):
			im_correlated[i,j] = np.sum(im[i-k:i+k+1,j-k:j+k+1] * kernel)
	#removes the undefined pixels i.e outer layers according to the size of kernel
	im_correlated_cropped = im_correlated[k:im.shape[0]-k, k:im.shape[1]-k]
	return im_correlated_cropped

def pad_image(im,no_of_layers):
	#Padding Type used below is replicate
	left_pad = im[0:im.shape[0],0][:,np.newaxis]
	right_pad = im[0:im.shape[0],-1][:,np.newaxis]
	im = np.hstack([left_pad,im,right_pad])
	top_pad = im[0, 0:im.shape[1]]
	bottom_pad = im[-1, 0:im.shape[1]]
	im = np.vstack([top_pad,im,bottom_pad])
	return im


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
	return g_mag,g_dir




def orientation_binning(g_mag,g_dir,cell_size,no_of_bins,binning_type='unsigned'):
	rows = g_mag.shape[0]
	columns = g_mag.shape[1]
	descriptors = np.zeros([int((rows/cell_size)*no_of_bins),int(columns/cell_size)])
	if(binning_type.lower()=='unsigned'):
		max_theta = 180		
	bin_size = max_theta/no_of_bins	
	bin_centers = np.arange(no_of_bins)*bin_size
	for i in range(0,rows,cell_size):
		for j in range(0,columns,cell_size):
		#histogram of cell
			histogram = np.zeros(no_of_bins)[:,np.newaxis]
			for m in range(i,i+cell_size):
				for n in range(j,j+cell_size):
					theta = g_dir[m,n]
					magnitude = g_mag[m,n]
					left_bin = int(floor(theta/bin_size))
					right_bin = int((ceil(theta/bin_size)))
					if(left_bin == right_bin):
						left_bin = left_bin%no_of_bins
						histogram[left_bin]+=magnitude
					else:
						right_bin = int(right_bin%no_of_bins)
						if(right_bin == 0):
							left_bin_vote = abs(theta - max_theta)/bin_size * magnitude
						else:
							left_bin_vote = abs(theta - bin_centers[right_bin])/bin_size * magnitude
						right_bin_vote = abs(theta - bin_centers[left_bin])/bin_size * magnitude
						histogram[left_bin]+=left_bin_vote
						histogram[right_bin]+=right_bin_vote
			k = int(i/cell_size*no_of_bins)
			l = int(j/cell_size)
			descriptors[k:k+no_of_bins,l][:,np.newaxis] = histogram
	return descriptors

#block size - 2 * 2 (in terms of cells)
#stride - 2 (in terms of cells)
def block_normalization(descriptors,block_size,stride,no_of_bins):
	rows = descriptors.shape[0]
	columns = descriptors.shape[1]
	v_stride = no_of_bins * stride
	v_size = block_size*no_of_bins
	feature_vector_rows = int(((rows - v_size)/v_stride + 1)*36)
	feature_vector_columns = int(columns - block_size + 1)
	feature_vector = np.zeros([feature_vector_rows,feature_vector_columns])
	for i in range(0,rows- v_size +1,v_stride):
		for j in range(0,columns-block_size +1,stride):
			block = descriptors[i:i+v_size, j:j+block_size].copy()
			# L2- Norm
			block = block/np.sqrt(np.sum(np.square(block))+ 0.001)
			k=int(i/no_of_bins*36)
			l=j
			feature_vector[k:k+36,l][:,np.newaxis] = block.flatten()[:,np.newaxis]
	return feature_vector

def flatten_HOG(feature_vector,block_size,no_of_bins):
	features = np.array([])
	rows = feature_vector.shape[0]
	columns = feature_vector.shape[1]
	v_stride = block_size**2 * no_of_bins
	for i in range(0,rows, v_stride ):
		for j in range(0, columns):
			x = feature_vector[i:i+v_stride, j]
			features = np.hstack([features,x])
	return features

# generates feature descriptor for slidng window
def HOG(im, cell_size, block_size, stride, no_of_bins, binning_type):
	im = pad_image(im,no_of_layers = 1)
	g_mag,g_dir = gradient(im,'prewitt')
	g_mag = np.round(((g_mag)/(sqrt(2)*765))*255)
	g_dir = g_dir/pi*180
	g_dir[g_dir<0] = 180 - np.abs(g_dir[g_dir<0])
	descriptors = orientation_binning(g_mag,g_dir,cell_size,no_of_bins,binning_type)
	feature_vector = block_normalization(descriptors,block_size,stride,no_of_bins)
	return feature_vector




def main():
	im = np.array(Image.open('HOG.jpeg').convert('L')).astype('float')
	print(np.shape(im))
	im = im[16:im.shape[0]-16,16:im.shape[1]-16]
	print(np.shape(im))
	feature_vector = HOG(im,8,2,1,9,'unsigned')
	print(np.shape(feature_vector))
	print(type(feature_vector))
	features = flatten_HOG(feature_vector,2,9)
	print(np.shape(features))
	print(type(features))



if __name__ == '__main__':
	main()