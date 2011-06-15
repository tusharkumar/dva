# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com


# Binning using Vector Quantization

# Input:
#   - List/Iterator of elements (multi-dim elements allowed)
#   - Stopping criteria:
#       + max number of classes
#       + distortion bounds (needed??)
#
# Output:
#   - List of class centroids
#   - num members in each class
#   - mean, std (distortion) in each class

# TO DO
# - Selectively split (instead of N = 2 *N in each step), based on comparing
#    distortion of each region with std/N (or std/2*N for safety margin)


import math
def vector_add(v1, v2):
	for i in range(len(v1)):
		v1[i] += v2[i]
	return v1

def vector_sub(v1, v2):
	for i in range(len(v1)):
		v1[i] -= v2[i]
	return v1

def div_vector(v, d):
	float_d = float(d)
	for i in range(len(v)):
		v[i] /= float_d
	return v

def scale_vector(v, s):
	for i in range(len(v)):
		v[i] *= s
	return v

def sq_norm(v):
	res = 0
	for i in range(len(v)):
		res += (v[i]) ** 2
	return res

def sq_dist(v1, v2):
	v_res = vector_sub(v1[:], v2)
	return sq_norm(v_res)


epsilon = 0.1
def vector_quantize(data_items, max_regions = None, CoV_thresh = None):
	Container_Class = type('Container_Class', (type(data_items), ), {})

	num_items = len(data_items)
	if num_items > 0:
		Item_Class = type('Item_Class', (type(data_items[0]), ), {})
		zero_item = data_items[0][:]
		for i in range(len(zero_item)):
			zero_item[i] = 0

	def container_vector_sum(item_list):
		cvs = zero_item[:]
		for item in item_list:
			vector_add(cvs, item)
		return cvs

	def container_vector_average(item_list):
		if len(item_list) > 0:
			return div_vector( container_vector_sum(item_list), float(len(item_list)) )
		else:
			return zero_item[:]

	def map_iter(func, item_list):
		for item in item_list:
			yield func(item)

	def container_distortion_sum(item_list, centroid):
		return sum( map_iter( lambda x: sq_dist(x, centroid), item_list) )

	c_init = container_vector_average(data_items)
	prev_distortion = container_distortion_sum(data_items, c_init)

	num_dims = None
	all_dim_mean = None

	if num_items > 0:
		num_dims = len(data_items[0])
		all_dim_mean = sum(c_init) / float(num_dims)

	cov = None
	Regions = [ {'centroid' : c_init, 'distortion' : prev_distortion, 'item_list' : data_items} ]
	while(True):
		print "Iteration with N = ", len(Regions)
		#print "  Regions = ", Regions

		if max_regions != None and len(Regions) >= max_regions:
			break

		# Split
		N = len(Regions)
		Regions.extend(N * [None]) # double the number of Regions
		for n in range(N, 2*N):
			Regions[n] = {}

		for n in range(N):
			orig_centroid = Regions[n]['centroid']
			Regions[n]['centroid'] = scale_vector( orig_centroid[:], (1+epsilon) )
			Regions[n]['distortion'] = 0
			Regions[n]['item_list'] = Container_Class()

			Regions[N+n]['centroid'] = scale_vector( orig_centroid[:], (1-epsilon) )
			Regions[N+n]['distortion'] = 0
			Regions[N+n]['item_list'] = Container_Class()

		N = len(Regions) #double N

		# Re-classify
		for item in data_items:
			min_sq_dist = None
			min_index = None
			for n in range(N):
				sq_dist_n = sq_dist(Regions[n]['centroid'], item)
				if min_sq_dist == None or min_sq_dist > sq_dist_n:
					min_sq_dist = sq_dist_n
					min_index = n

			Regions[min_index]['item_list'].append(item)

		#Update Centroids
		for n in range(N):
			Regions[n]['centroid'] = container_vector_average( Regions[n]['item_list'] )

		#Update Distortions:
		new_distortion = 0
		for n in range(N):
			Regions[n]['distortion'] = container_distortion_sum( \
					Regions[n]['item_list'], Regions[n]['centroid'] )
			new_distortion += Regions[n]['distortion']

		variance = (1.0 / num_items) * (1.0 / num_dims) * new_distortion
		std = math.sqrt(variance)
		cov = std / all_dim_mean
		print " $$ cov = ", cov

		print " ### prev_distortion = ", prev_distortion, " new_distortion = ", new_distortion
		if new_distortion < epsilon or (prev_distortion - new_distortion) / prev_distortion < epsilon:
			break

		if CoV_thresh != None and cov <= CoV_thresh:
			print "Breaking: cov = ", cov, " is <= CoV_thresh = ", CoV_thresh
			break

		prev_distortion = new_distortion


	success = True
	if CoV_thresh != None:
		if cov == None or cov > CoV_thresh:
			success = False

	NonEmpty_Regions = [reg for reg in Regions if len(reg['item_list']) > 0]
	return (success, cov, NonEmpty_Regions)
