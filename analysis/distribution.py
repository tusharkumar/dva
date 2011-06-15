#!/usr/bin/python

# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com


import math


#erf code from:
#  http://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/

def erf(x):
	# constants
	a1 =  0.254829592
	a2 = -0.284496736
	a3 =  1.421413741
	a4 = -1.453152027
	a5 =  1.061405429
	p  =  0.3275911

	# Save the sign of x
	sign = 1
	if x < 0:
		sign = -1
	x = abs(x)

	# A & S 7.1.26
	t = 1.0/(1.0 + p*x)
	y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)

	return sign*y


sqrt_of_2 = math.sqrt(2)
def gaussian_cdf(mu, sigma, x):
	return 0.5 * ( 1.0 + erf((x - mu) / (sigma * sqrt_of_2)) )


####

class Distribution:
	def __init__(self, mean_std_count_list = None, gaussian_mixture_range = (None, None)):
		'''gaussian_mixture_range is inferred from mean_std_count_list if not specified'''

		self.mean_std_count_list = []
			#each element is a tuple of form (mean, std, count),
			# where count represents the number of sample points over which mean, std is computed
			#Required invariant:
			# - elements are sorted in ascending order of mean field

		self.gaussian_mixture_range = (None, None)
			#range of x values outside which the gaussian mixture density becomes insignificant

		if mean_std_count_list != None:
			assert len(mean_std_count_list) > 0
			self.mean_std_count_list = mean_std_count_list
			self.infer_significant_gaussian_mixture_range()

		if gaussian_mixture_range[0] != None:
			self.gaussian_mixture_range = (gaussian_mixture_range[0], self.gaussian_mixture_range[1])
		if gaussian_mixture_range[1] != None:
			self.gaussian_mixture_range = (self.gaussian_mixture_range[0], gaussian_mixture_range[1])

	def clone(self):
		return Distribution( self.mean_std_count_list, self.gaussian_mixture_range )

	def infer_significant_gaussian_mixture_range(self):
		'''infer from min and max of values (mean_i - k * std_i, mean_i + k * std_i), assuming Gaussian Mixture Density'''
		k = 3.0 # Assume: k standard deviations contain almost all of the probability
		lower_bounds = [(mean - k * std) for mean, std, count in self.mean_std_count_list]
		upper_bounds = [(mean + k * std) for mean, std, count in self.mean_std_count_list]

		self.gaussian_mixture_range = ( min(lower_bounds), max(upper_bounds) )

	def get_combined_mean_std_count(self):
		'''Distribution Independent'''
		combined_count = sum( [count for (mean, std, count) in self.mean_std_count_list] )
		if combined_count == 0:
			return (0.0, 0.0, 0)

		combined_mean = sum( [mean * count for (mean, std, count) in self.mean_std_count_list] ) / combined_count
		combined_var = sum( [ (std * std + (mean - combined_mean) ** 2) * count for (mean, std, count) in self.mean_std_count_list] ) / combined_count
		combined_std = math.sqrt(combined_var)

		return (combined_mean, combined_std, combined_count)

	def evaluate_gaussian_mixture_cdf(self, x):
		assert len(self.mean_std_count_list) > 0

		combined_count = sum( [count for (mean, std, count) in self.mean_std_count_list] )
		evaluated_cdf_val = sum( [float(count)/combined_count * gaussian_cdf(mean, std, x) for mean, std, count in self.mean_std_count_list] )
		return evaluated_cdf_val

	#def __repr__(self):
	#	return "Distribution( mean_std_count_list = %s, gaussian_mixture_range = %s )" % (self.mean_std_count_list, self.gaussian_mixture_range)

	def __repr__(self):
		return "%s" % (self.mean_std_count_list, )


def kolmogorov_smirnov_difference(dist1, dist2):
	print "Invoked: kolmogorov_smirnov_difference(): dist1 = %s, dist2 = %s" % (dist1, dist2)
	assert len(dist1.mean_std_count_list) > 0
	assert len(dist2.mean_std_count_list) > 0

	num_mixtures = max( len(dist1.mean_std_count_list), len(dist2.mean_std_count_list) )
	num_comparison_points = num_mixtures * 100

	comparison_range_lower_bound = min(dist1.gaussian_mixture_range[0], dist2.gaussian_mixture_range[0])
	comparison_range_upper_bound = max(dist1.gaussian_mixture_range[1], dist2.gaussian_mixture_range[1])

	step_size = (comparison_range_upper_bound - comparison_range_lower_bound) / float(num_comparison_points)

	D = 0.0
	x = comparison_range_lower_bound
	while x <= comparison_range_upper_bound:
		cdf1_at_x = dist1.evaluate_gaussian_mixture_cdf(x)
		cdf2_at_x = dist2.evaluate_gaussian_mixture_cdf(x)
		difference = abs(cdf1_at_x - cdf2_at_x)
		if difference > D:
			D = difference
		x += step_size

	print "  D =", D
	return D

def produce_reduced_distribution(dist, close_enough_D = 0.10):
	working_list_of_gaussians = dist.mean_std_count_list[:] #make copy, dist is to be treated as read-only
		#new merged gaussians added to the end, consumed gaussian entries crossed out by replacing with None

	merge_candidates_list = []
	for i in range( len(working_list_of_gaussians) ):
		for j in range(i+1, len(working_list_of_gaussians)):
			D_i_j = kolmogorov_smirnov_difference( Distribution( [working_list_of_gaussians[i]] ), Distribution( [working_list_of_gaussians[j]] ) )
			if D_i_j <= close_enough_D:
				merge_candidates_list.append( (i, j, D_i_j) )

	merge_candidates_list.sort( lambda x, y: cmp(x[2], y[2]) ) #sort in ascending D's

	#process merge_candidates_list:
	while len(merge_candidates_list) > 0:
		(i, j, D) = merge_candidates_list.pop(0)
		assert D <= close_enough_D

		if working_list_of_gaussians[i] == None or working_list_of_gaussians[j] == None:
			continue #already merged with something

		# Now, neither is merged as yet

		#get merged distribution of i and j
		merged_gaussian = Distribution( [ working_list_of_gaussians[i], working_list_of_gaussians[j] ] ).get_combined_mean_std_count()

		k = len(working_list_of_gaussians)
		working_list_of_gaussians.append( merged_gaussian )

		#cross out i and j
		working_list_of_gaussians[i] = None
		working_list_of_gaussians[j] = None

		#add new candidates to merge_candidates_list
		merged_dist = Distribution( [merged_gaussian] )
		for q in range( k ): #compare with everything but k
			if working_list_of_gaussians[q] == None:
				continue

			D_q_k = kolmogorov_smirnov_difference( Distribution( [working_list_of_gaussians[q]] ), merged_dist )
			if D_q_k <= close_enough_D:
				merge_candidates_list.append( (q, k, D_q_k) )

		merge_candidates_list.sort( lambda x, y: cmp(x[2], y[2]) ) #sort in ascending D's

	final_gaussians = [gaussian for gaussian in working_list_of_gaussians if gaussian != None]

	return Distribution( final_gaussians )

####

def print_pdfs_over_range( list_of_distributions ):
	x = -15.0
	while x < 15.0:
		print x,
		for dist in list_of_distributions:
			derivative = (dist.evaluate_gaussian_mixture_cdf(x) - dist.evaluate_gaussian_mixture_cdf(x-0.001))/0.001
			print derivative,
		print
		x += 0.1

def run_test_reduction():
	dist1 = Distribution( [(-5.0, 1.0, 50), (5.0, 1.0, 100), (8.0, 1.0, 250)] )
	print "dist1 =", dist1

	reduced_dist = produce_reduced_distribution(dist1, 0.9)
	print "reduced_dist =", reduced_dist

	print_pdfs_over_range( [dist1, reduced_dist] )


####

class BinnedDistribution:
	def __init__(self):
		self.bins = []
			# list of tuples: (key, dist) where key is a user-defined entity that is unique within self.bins, and dist is an instance of Distribution
			# NOTE: add or remove tuples only via add_bin() or remove_bin()

	def add_bin(self, add_key, add_dist = None):
		for key, dist in self.bins:
			if key == add_key:
				sys.exit( "BinnedDistribution: add_bin(): ERROR: key being added is already present. key = %s" % (key, ) )

		if add_dist == None:
			add_dist = Distribution() #empty distribution

		self.bins.append( (add_key, add_dist) )

	def get_index_of_key_in_bins(self, search_key):
		for i, (key, dist) in enumerate(self.bins):
			if key ==search_key:
				return i
		return None #not found

	def remove_bin(self, remove_key):
		remove_loc = self.get_index_of_key_in_bins(remove_key)
		if remove_loc == None:
			sys.exit( "BinnedDistribution: remove_bin(): ERROR: key being removed is not present. key = %s" % (remove_key, ) )
		return self.bins.pop(remove_loc)


	def compute_distortion(self):
		distortion = 0.0
		combined_count = 0
		for key, dist in self.bins:
			mean, std, count = dist.get_combined_mean_std_count()
			distortion += std * std * count #bin-variance * weight-of-bin
			combined_count += count
		assert combined_count > 0
		distortion = distortion / combined_count
		return distortion


#FIXME: Questions
# 1) What is the underlying optimization goal or correctness "meaningfullness" constraint for merging contributor-distributions?
#  Possibilities:
#    - minimize std for each bin in the contributor-distribution ==> distortion bounding is a good approach for this
# 
#  Issue: What does a 10% tolerance on merging mean and std (to compute D_KS_max) correspond to when comparing contribution-distributions?
#    - D-statistic not directly usable:
#         + bin-labels correspond to contributors i.e. histogram => no ordering
#         + 


if __name__ == "__main__":
	run_test_reduction()
else:
	print "~~ IMPORTED distribution ~~"


