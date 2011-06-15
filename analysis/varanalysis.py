#!/usr/bin/python

# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com


###

import sys

import common

################################################
####### CONSTRUCT PROFILE REPRESENTATION #######
################################################

import profile

def set_global_profile_representation(given_profile_repr):
	global profile_repr
	profile_repr = given_profile_repr

	global func_id_to_name, map_func_name_to_id, func_id_to_info, map_func_name_to_loop_hier, \
				profile_cet, max_count, min_significant_exec_time, map_called_NULL_FUNC_artificial_lexical_id
	func_id_to_name            = profile_repr.func_id_to_name
	map_func_name_to_id        = profile_repr.map_func_name_to_id
	func_id_to_info            = profile_repr.func_id_to_info
	map_func_name_to_loop_hier = profile_repr.map_func_name_to_loop_hier
	profile_cet                = profile_repr.profile_cet
	max_count                  = profile_repr.max_count
	min_significant_exec_time  = profile_repr.min_significant_exec_time
	map_called_NULL_FUNC_artificial_lexical_id = profile_repr.map_called_NULL_FUNC_artificial_lexical_id


def del_global_profile_representation():
	global profile_repr

	global func_id_to_name, map_func_name_to_id, func_id_to_info, map_func_name_to_loop_hier, \
				profile_cet, max_count, min_significant_exec_time, map_called_NULL_FUNC_artificial_lexical_id

	del profile_repr

	del func_id_to_name, map_func_name_to_id, func_id_to_info, map_func_name_to_loop_hier, \
				profile_cet, max_count, min_significant_exec_time, map_called_NULL_FUNC_artificial_lexical_id


profile_file_name = "profile.dump"
enable_cross_covar_analysis = True
def read_profile_representation():
	new_profile_repr = profile.ProfileRepresentation()
	new_profile_repr.read("func_map.dump", "func_loop_hier.dump", profile_file_name, enable_cross_covar_analysis)

	return new_profile_repr


#set_global_profile_representation( read_profile_representation() )




############################################################
######## DEFINE REPEATABLE VARIANCE ANALYSES ON CET ########
############################################################

import ccanalysis


################ Variance Tagging Analysis ################

# Chebyshev Inequality: Pr[|X-u| > k * s] < 1/k^2 (escape-bounds probability)
#  CoV = s/u
# Window * u = k * s
#    => CoV = Window/k

import math
escape_bounds_probability = 0.04
k = math.sqrt(1.0/escape_bounds_probability)

window_bound_high_var = 2    #variation exceeds 'window_bound_high_var' * 'mean' => high-var
window_bound_low_var  = 0.1  #variation is below 'window_bound_low_var' * 'mean' => low-var

cov_high_threshold = window_bound_high_var / k  #0.4 # PLDI 2008 change #0.3
cov_low_threshold  = window_bound_low_var  / k  #0.02 # PLDI 2008 change #0.01

def covar_cmp(x, y): # for sort-descending
	ax = abs(x[0])
	ay = abs(y[0])
	if ax > ay: return -1
	elif ax < ay: return 1
	else: return 0

def get_sorted_covar_terms(func_node, threshold = 0.0):
	term_list = []
	for i in range(len(func_node.cross_sq_err_sum)):
		for j in range(0, i):
			term_list.append( (func_node.cross_sq_err_sum[i][j] * 2, i, j) ) # off-diagonal terms
		term_list.append( (func_node.cross_sq_err_sum[i][i], i, i) ) # on-diagonal terms
	
	term_list2 = [term for term in term_list if abs(term[0]) > threshold]
	term_list2.sort(covar_cmp) # sort on covar values

	return term_list2

def get_sorted_child_var_terms(func_node, threshold = 0.0):
	term_list = []
	for i in range(len(func_node.cross_sq_err_sum)):
		term_list.append( (func_node.cross_sq_err_sum[i][i], i, i) ) # on-diagonal terms
	
	term_list2 = [term for term in term_list if term[0] > threshold]
	term_list2.sort(covar_cmp) # sort on var values

	return term_list2

# Fraction compensate *against* node's variance being high/medium/low
covar_fraction_for_high = 0.10
covar_fraction_for_medium = 0.30
covar_fraction_for_low = 0.50

covar_exposer_factor = 5

######## TO DO: FIXES
# 2) ADD DFS for traversal: low recursion limit in Python

min_invoke_count_needed_for_pattern = 6
def tag_variant_nodes_recursive(func_node):
	if func_node.total_count < min_significant_exec_time:
		return

	mean = float(func_node.total_count) / float(func_node.invoke_count)
	std = math.sqrt( func_node.sq_err_sum / float(func_node.invoke_count) )
	cov = std / mean

	func_node.analysis["variant"] = None
	func_id = func_node.func_id

	if func_node.invoke_count >= min_invoke_count_needed_for_pattern:
		if cov > cov_high_threshold:
			func_node.analysis["variant"] = "high"
			#print "Call-Chain-Context has 'high' cov =", cov, "with mean =", mean, " : ", func_node.get_call_chain_context()

		if cov < cov_low_threshold:
			func_node.analysis["variant"] = "low"
			#print "Call-Chain-Context has 'low' cov =", cov, "with mean =", mean, " : ", func_node.get_call_chain_context()

		if func_node.analysis["variant"] == "high":
			covar_term_threshold = func_node.sq_err_sum * covar_fraction_for_high
		elif func_node.analysis["variant"] == None:
			covar_term_threshold = func_node.sq_err_sum * covar_fraction_for_medium
		elif func_node.analysis["variant"] == "low":
			covar_term_threshold = func_node.sq_err_sum * covar_fraction_for_low
		else:
			sys.exit("tag_variant_nodes_recursive(): ERROR: unknown value for 'variant' = " + str(func_node.analysis["variant"]))

		covar_term_list = get_sorted_covar_terms(func_node, covar_term_threshold)
		func_node.analysis["covar_cause"] = covar_term_list
		#print "   covar_cause = ", covar_term_list

		cross_covar_term_list = [(val, i, j) for (val, i, j) in covar_term_list if i != j] 
		cross_abs_sum = sum( [abs(val) for (val, i, j) in cross_covar_term_list] )

		if cross_abs_sum > covar_term_threshold * covar_exposer_factor:
			func_node.analysis["cross_covar_exposer"] = True
			func_node.analysis["cross_covar_term_list"] = cross_covar_term_list
			#print "Call-Chain-Context is EXPOSER of cross-covariance with sum = ", cross_abs_sum, " : ", func_node.get_call_chain_context()
		else:
			func_node.analysis["cross_covar_exposer"] = False


	for c in func_node.funcs_called:
		if c != None:
			tag_variant_nodes_recursive(c)


### Variance Tagging Helpers

def high_variant_match_criteria_func(func_node):
	if func_node.analysis.has_key("variant"):
		if func_node.analysis["variant"] == "high":
			return True
	return False

def low_variant_match_criteria_func(func_node):
	if func_node.analysis.has_key("variant"):
		if func_node.analysis["variant"] == "low":
			return True
	return False

def cross_covar_exposer_match_criteria_func(func_node):
	if func_node.analysis.has_key("cross_covar_exposer"):
		return func_node.analysis["cross_covar_exposer"]
	return False

def user_spec_scope_match_criteria_helper(func_node, user_scope):
	'''user-scope: [(FN, LN) .. (F1, L1)]
		Checks whether the call-context subsequence (not sub-string, contiguity is not required) (FN, LN) .. (F1, L1)
		occurs in func_node's call-context. F1 is outermost part of scope, FN is innermost part of scope.
	LN .. L1 represent the corresponding func_lexical_ids. Li = None implies a "don't-care" for matching Fi's
	  func_lexical_id in the scope.
	user_scope_id: ID to allow compact/rapid identification of user-scope
	'''
	cc = func_node.get_call_chain_context()
	index = 0
	for cc_func_name, cc_func_lexical_id in cc:
		if index < len(user_scope):
			if cc_func_name == user_scope[index][0] \
					and (user_scope[index][1] == None or user_scope[index][1] == cc_func_lexical_id):
				index += 1
	if index == len(user_scope): #user-scope sub-sequence occurs
		return True
	return False

class user_spec_high_var_scope_match_criteria_func_class:
	def __init__(self, user_scope, user_scope_id):
		self.user_scope = user_scope
		self.user_scope_id = user_scope_id
			#user_scope_id provides a compact representation of the 'user-scope' for annotating func_nodes
	
	def match_criteria_func(self, func_node):
		matches = high_variant_match_criteria_func(func_node) \
				and user_spec_scope_match_criteria_helper(func_node, self.user_scope)
		if matches:
			if not func_node.analysis.has_key("user_scope_id_list"):
				func_node.analysis["user_scope_id_list"] = []
			if not self.user_scope_id in func_node.analysis["user_scope_id_list"]:
				func_node.analysis["user_scope_id_list"].append(self.user_scope_id)
		return matches


################ Pattern Similarity Tree Helpers ################

### high-variant
high_variant_mean_epsilon = 0.01
high_variant_cov_epsilon = 0.01
def high_variant_sm_func(p1, p2): #compare similarity of two high-variant patterns with identical call-chains
	func_node1 = p1[0]
	func_node2 = p2[0]

	mean1 = float(func_node1.total_count) / float(func_node1.invoke_count)
	std1 = math.sqrt( func_node1.sq_err_sum / float(func_node1.invoke_count) )
	cov1 = std1 / mean1

	mean2 = float(func_node2.total_count) / float(func_node2.invoke_count)
	std2 = math.sqrt( func_node2.sq_err_sum / float(func_node2.invoke_count) )
	cov2 = std2 / mean2

	if abs(mean1 - mean2) < high_variant_mean_epsilon * mean1: # means close enough
		if abs(cov1 - cov2) < high_variant_cov_epsilon * cov1: # covs close enough as well
			return True

	return False

### low-variant
low_variant_mean_epsilon = 0.01
def low_variant_sm_func(p1, p2): #compare similarity of two low-variant patterns with identical call-chains
	func_node1 = p1[0]
	func_node2 = p2[0]

	mean1 = float(func_node1.total_count) / float(func_node1.invoke_count)
	mean2 = float(func_node2.total_count) / float(func_node2.invoke_count)

	if abs(mean1 - mean2) < low_variant_mean_epsilon * mean1: # means close enough
		return True

	return False

### cross-covar-exposer
def get_cross_covar_direction_list(cross_covar_term_list):
	cross_covar_direction_list = [(cmp(val, 0.0), i, j) for (val, i, j) in cross_covar_term_list]
	max_index = -1
	for (dir, i, j) in cross_covar_direction_list:
		if i > max_index:
			max_index = i
		if j > max_index:
			max_index = j
	max_index += 1
		
	cross_covar_direction_list.sort(lambda x, y: (x[1] * max_index + x[2]) - (y[1] * max_index + y[2])) #sort on i, then j
		#sorting done to ensure fixed order of terms: useful for direct comparison of direction_lists

	return cross_covar_direction_list

def cross_covar_exposer_sm_func(p1, p2):
	func_node1 = p1[0]
	func_node2 = p2[0]

	dir_list1 = get_cross_covar_direction_list( func_node1.analysis["cross_covar_term_list"] )
	dir_list2 = get_cross_covar_direction_list( func_node2.analysis["cross_covar_term_list"] )

	if dir_list1 == dir_list2:
		return True

	return False



################ Graph Dumping Helpers ################

def significant_subtree_cutoff_test_func(func_node):
	if func_node.total_count >= min_significant_exec_time:
		return True
	return False


def node_decorator_func(func_node):
	label_parm = '{%s: %s' % (func_node.node_index, func_node.func_name)
	display_parms = ""

	mean = float(func_node.total_count) / float(func_node.invoke_count)
	std = math.sqrt( func_node.sq_err_sum / float(func_node.invoke_count) )
	cov = std / mean
	label_parm += '| invoke=%s total=%s sq_err_sum=%.2f' % (func_node.invoke_count, func_node.total_count, func_node.sq_err_sum)
	label_parm += '| mean=%.2f std=%.2f CoV=%.2f' % (mean, std, cov)

	if func_node in best_high_var_pattern_tagged_nodes:
		display_parms += " fillcolor=pink"
	elif func_node in best_low_var_pattern_tagged_nodes:
		display_parms += " fillcolor=lightblue"
	else:
		display_parms += " fillcolor=white"

	if func_node in best_cross_covar_exposer_pattern_tagged_nodes:
		display_parms += " color=crimson"
	else:
		display_parms += " color=lightgrey"

	label_parm += '}'

	decorator = 'label="%s" %s' % (label_parm, display_parms)
	return decorator



def clamp_at_top_fraction_of_func_nodes(func_node_list, top_fraction):
	func_node_list.sort(lambda x, y: cmp(y.total_count, x.total_count)) #descending order
	top_exec_time_threshold = sum( [fn.total_count for fn in func_node_list] ) * top_fraction

	for i, fn in enumerate(func_node_list):
		if fn.total_count < top_exec_time_threshold:
			func_node_list[i:] = []
			break







################ Pattern Formation Helpers ################

def merge_sorted_bins(inplace_bins, extra_bins):
	i, j = 0, 0
	while i < len(inplace_bins) and j < len(extra_bins):
		i_centroid = inplace_bins[i][0]
		j_centroid = extra_bins[j][0]

		if j_centroid <= i_centroid:
			inplace_bins.insert(i, extra_bins[j])
			j += 1
		else:
			i += 1

	if i == len(inplace_bins):
		inplace_bins.extend(extra_bins[j:])
		


def get_merged_mean_cov(mean_count_cov_1, mean_count_cov_2):
	(mean1, count1, cov1) = mean_count_cov_1
	(mean2, count2, cov2) = mean_count_cov_2

	std1 = cov1 * mean1
	std2 = cov2 * mean2
	
	var1 = std1 ** 2
	var2 = std2 ** 2

	mean_merged = (mean1 * count1 + mean2 * count2) / float(count1 + count2)
	var_merged = (count1 * (var1 + (mean1 - mean_merged) ** 2) + count2 * (var2 + (mean2 - mean_merged) ** 2)) \
					/ float(count1 + count2)
	std_merged = math.sqrt(var_merged)
	if mean_merged > 0:
		cov_merged = std_merged / mean_merged
	else:
		if std_merged > 0:
			cov_merged = None
		else:
			cov_merged = 0.0 # both mean and std == 0.0

	return (mean_merged, count1 + count2, cov_merged)

def compress_sorted_bins(inplace_bins, cov_threshold):
	i = 0
	while i < len(inplace_bins) - 1:
		merged_bin = get_merged_mean_cov(inplace_bins[i], inplace_bins[i+1])
		cov_merged = merged_bin[2]

		if cov_merged <= cov_threshold:
			inplace_bins[i] = merged_bin
			del inplace_bins[i+1]
		else:
			i += 1


import linearpattern

### high-variant
def high_variant_setup_new_pattern_stats(func_node, new_pattern):
	new_pattern.exec_time_weight = func_node.total_count
	new_pattern.pred_behavior["variant"] = "high"
	new_pattern.pred_behavior["sq_err_sum"] = func_node.sq_err_sum

	new_pattern.pred_behavior["mean"] = float(func_node.total_count) / float(func_node.invoke_count)
	new_pattern.pred_behavior["std"] = math.sqrt( func_node.sq_err_sum / float(func_node.invoke_count) )
	new_pattern.pred_behavior["cov"] = new_pattern.pred_behavior["std"] / new_pattern.pred_behavior["mean"]

	if func_node.analysis.has_key("high_var_bins"):
		new_pattern.pred_behavior["high_var_bins"] = func_node.analysis["high_var_bins"]

def high_variant_combine_stats_with_pattern(func_node, pattern_with_matching_call_chain):
	pattern_with_matching_call_chain.exec_time_weight += func_node.total_count
	pattern_with_matching_call_chain.pred_behavior["sq_err_sum"] += func_node.sq_err_sum

	if pattern_with_matching_call_chain.pred_behavior.has_key("high_var_bins"):
		if func_node.analysis.has_key("high_var_bins"):
			merge_sorted_bins(pattern_with_matching_call_chain.pred_behavior["high_var_bins"], \
							func_node.analysis["high_var_bins"])
			compress_sorted_bins(pattern_with_matching_call_chain.pred_behavior["high_var_bins"], cov_low_threshold)

		else:
			del pattern_with_matching_call_chain.pred_behavior["high_var_bins"]


	#Assume that pattern mean, std and cov don't change much

	return True #always combine



### low-variant
#low_variant_mean_epsilon = 0.01
def low_variant_setup_new_pattern_stats(func_node, new_pattern):
	new_pattern.exec_time_weight = func_node.total_count
	new_pattern.pred_behavior["variant"] = "low"
	new_pattern.pred_behavior["mean"] = float(func_node.total_count) / float(func_node.invoke_count)

def low_variant_combine_stats_with_pattern(func_node, pattern_with_matching_call_chain):
	func_node_mean = float(func_node.total_count)/ float(func_node.invoke_count)

	if abs(func_node_mean - pattern_with_matching_call_chain.pred_behavior["mean"]) < low_variant_mean_epsilon * func_node_mean:
		pattern_with_matching_call_chain.exec_time_weight += func_node.total_count
		#Assume that pattern mean doesn't change much
		return True #ok to combine

	return False #don't combine


### cross-covar-exposer
def map_child_lexical_id_to_covar_term_name(func_node, lexical_id):
	if lexical_id == len(func_node.funcs_called): #pure parent exec-time
		return '*'
	else:
		return func_node.funcs_called[lexical_id].func_name

def cross_covar_exposer_setup_new_pattern_stats(func_node, new_pattern):
	new_pattern.exec_time_weight = func_node.total_count
	new_pattern.pred_behavior["cross_covar_exposer"] = True
	cross_covar_direction_list = get_cross_covar_direction_list( func_node.analysis["cross_covar_term_list"] )
	new_pattern.pred_behavior["cross_covar_direction_list"] = cross_covar_direction_list

	new_pattern.pred_behavior["lexical_id_to_child_func_name_map"] = {}
	for val, i, j in cross_covar_direction_list:
		new_pattern.pred_behavior["lexical_id_to_child_func_name_map"][i] = map_child_lexical_id_to_covar_term_name(func_node, i)
		new_pattern.pred_behavior["lexical_id_to_child_func_name_map"][j] = map_child_lexical_id_to_covar_term_name(func_node, j)
		
	if func_node.analysis.has_key("cross_covar_bins"):
		new_pattern.pred_behavior["cross_covar_bins"] = func_node.analysis["cross_covar_bins"]

def cross_covar_exposer_combine_stats_with_pattern(func_node, pattern_with_matching_call_chain):
	func_node_cross_covar_direction_list = get_cross_covar_direction_list( func_node.analysis["cross_covar_term_list"] )

	if func_node_cross_covar_direction_list == pattern_with_matching_call_chain.pred_behavior["cross_covar_direction_list"]:
		pattern_with_matching_call_chain.exec_time_weight += func_node.total_count

		if pattern_with_matching_call_chain.pred_behavior.has_key("cross_covar_bins"):
			if func_node.analysis.has_key("cross_covar_bins"):
				pattern_with_matching_call_chain.pred_behavior["cross_covar_bins"] \
							.extend(func_node.analysis["cross_covar_bins"])

			else:
				del pattern_with_matching_call_chain.pred_behavior["cross_covar_bins"]

		return True

	return False


############################################
###### DEFINE DEMAND-DRIVEN PROFILING ######
############################################

################ Binning Analysis ################

def collect_binning_samples_on_node_visit(curr_node, count_diff, children_exec_counts):
	if curr_node.analysis.has_key("high_var_bins"): # Do High-Var binning
		if "samples" not in dir(curr_node):
			curr_node.samples = []
		curr_node.samples.append( [count_diff] )


	node_internal_exec_count = count_diff - sum(children_exec_counts)
	all_exec_counts = children_exec_counts + [node_internal_exec_count]

	if curr_node.analysis.has_key("cross_covar_bins"): # Do Cross-Covar binning
		if "cross_covar_samples" not in dir(curr_node):
			curr_node.cross_covar_samples = []
			curr_node.correlate_child_flag_list = [False] * (len(curr_node.funcs_called) + 1)
			for (val, i, j) in curr_node.analysis["cross_covar_term_list"]:
				curr_node.correlate_child_flag_list[i] = True
				curr_node.correlate_child_flag_list[j] = True
		curr_cross_sample = [ all_exec_counts[i] for i in range(len(all_exec_counts)) \
									if curr_node.correlate_child_flag_list[i] ]
		curr_node.cross_covar_samples.append( curr_cross_sample )


import vq

# Steps:
# 1) Construct list of high-var nodes to bin
# 2) Construct list of cross-covar nodes to correlate
#     - sublists of correlated nodes with cross-covar exposers
# 3) Make profile pass: read up relevant execution times
#   - simple sequence for high-var
#   - tuples for cross-covar exposers

def do_binning_profiling_and_analysis(high_var_pattern_tagged_nodes, cross_covar_exposer_pattern_tagged_nodes, \
		associated_profile_repr):
	'''Runs an additional profile pass on a previously constructed profile-representation,
		and attempts to bin execution-times on specified instances of FuncNodes.
		Binned execution times are annotated on FuncNode instances where binning succeeds.

		Parms:
			high_var_pattern_tagged_nodes = a list identifying high-variance func-nodes for binning,
			cross_covar_exposer_pattern_tagged_nodes = a list identifying cross-covar-exposer func-nodes for binning,
			associated_profile_repr = the profile-representation within which the identified func-nodes occur

		Resulting FuncNode annotations:
			High-variance func-nodes where binning succeeded:
				func_node.analysis["high_var_bins"] = list of bins
				func_node.analysis["high_var_binned_cov"] = cov_overall

			Cross-covar-exposer func-nodes where binning succeeded:
				func_node.analysis["cross_covar_bins"] = list of bins
				func_node.analysis["cross_covar_binned_cov"] = cov_overall

			In both cases, the following applies:
				- absence of annotations => func_node's execution times were not binnable
				- a 'bin' is a tuple of
					(mean-execution-time of bin, number of execution instances in this bin, cov within this bin)
					where mean-execution-time = scalar for high-variance nodes, and
					                 = a tuple of cross-correlated children execution times for cross-covar-exposer
				- cov_overall represents combined cov of all execution times of the func_node calculated around the
				    means of bins where each occured

	'''
	for func_node in high_var_pattern_tagged_nodes:
		func_node.analysis["high_var_bins"] = []

	for func_node in cross_covar_exposer_pattern_tagged_nodes:
		func_node.analysis["cross_covar_bins"] = []

	if associated_profile_repr.pass_number < 2 or associated_profile_repr.pass_number > 3:
		sys.exit("do_binning_profiling_and_analysis(): ERROR: pass_number must be 2 or 3 for this phase. Found pass_number = %s" % (associated_profile_repr.pass_number))

	if associated_profile_repr.pass_number == 3:
		associated_profile_repr.pass_number = 2

	profile.run_user_profile_pass(associated_profile_repr, collect_binning_samples_on_node_visit) #pass 3

	common.MARK_PHASE_START("VECTOR QUANTIZATION BINNING")

	max_high_variance_bins = 8
	for func_node in high_var_pattern_tagged_nodes:
		print "----#### Binning high-var", (func_node.func_name, func_node.func_lexical_id)
		(success, cov_overall, regions) = vq.vector_quantize(func_node.samples, \
											max_regions = max_high_variance_bins, \
											CoV_thresh = cov_low_threshold)
		del func_node.samples

		if success == False:
			print "  $$ Binning FAILED"
			del func_node.analysis["high_var_bins"]
			continue

		# Binning Succeeded
		bins = []
		for reg in regions:
			num_items_in_region = len(reg["item_list"])
			reg_centroid = reg["centroid"][0] # 1-dimensional data
			reg_variance = reg["distortion"] / float(num_items_in_region) # 1-dimensional data
			reg_std = math.sqrt(reg_variance)
			if reg_centroid > 0.0:
				reg_cov = reg_std / reg_centroid
			else:
				reg_cov = 0.0

			bins.append( (reg_centroid, num_items_in_region, reg_cov) )

		bins.sort(lambda x, y: cmp(x[0], y[0])) #sort on centroids
		func_node.analysis["high_var_bins"] = bins
		func_node.analysis["high_var_binned_cov"] = cov_overall
		print "   $$ Bins found for ", (func_node.func_name, func_node.func_lexical_id), " binned_cov = ", cov_overall, ":"
		print "     ", func_node.analysis["high_var_bins"]
		print

	max_cross_covar_bins_multiplier = 8
	cov_cross_covar_bins_threshold = (cov_low_threshold + cov_high_threshold) / 3.0
	for func_node in cross_covar_exposer_pattern_tagged_nodes:
		print "----#### Binning cross-covar", (func_node.func_name, func_node.func_lexical_id)
		print "          cross_covar_term_list = ", func_node.analysis["cross_covar_term_list"]
		num_dims = len(func_node.cross_covar_samples[0]) #assume atleast one, for node to have been tagged cross-covar at all
		(success, cov_overall, regions) = vq.vector_quantize(func_node.cross_covar_samples, \
											max_regions = max_cross_covar_bins_multiplier * num_dims, \
											CoV_thresh = cov_cross_covar_bins_threshold)
		del func_node.cross_covar_samples
		del func_node.correlate_child_flag_list

		if success == False:
			print "  $$ Binning FAILED"
			del func_node.analysis["cross_covar_bins"]
			continue

		# Binning Succeeded
		bins = []
		for reg in regions:
			num_items_in_region = len(reg["item_list"])
			reg_vq_codeword = reg["centroid"]
			reg_mean = sum(reg_vq_codeword) / float(num_dims)
			reg_variance = reg["distortion"] / float(num_items_in_region) / float(num_dims)
			reg_std = math.sqrt(reg_variance)
			if reg_mean > 0:
				reg_cov = reg_std / reg_mean
			else:
				if reg_std > 0:
					reg_cov = None
				else:
					reg_cov = 0.0 # both mean and std == 0.0

			bins.append( (reg_vq_codeword, num_items_in_region, reg_cov) )

		func_node.analysis["cross_covar_bins"] = bins
		func_node.analysis["cross_covar_binned_cov"] = cov_overall
		print "   $$ Bins found for ", (func_node.func_name, func_node.func_lexical_id), " binned_cov = ", cov_overall, ":"
		print "     ", func_node.analysis["cross_covar_bins"]
		print

	common.MARK_PHASE_COMPLETION("VECTOR QUANTIZATION BINNING")


		
		
		

###################################################
######## APPLY REPEATABLE ANALYSES ON CET ########
###################################################

class VarianceAnalysisCETResults:
	def __init__(
		self,
		significant_func_list                         = None,
		linear_pattern_vec                            = None,
		best_high_var_pattern_tagged_nodes            = None,
		best_low_var_pattern_tagged_nodes             = None,
		best_cross_covar_exposer_pattern_tagged_nodes = None
	):
		#FIXME: Add explanations for the following fields
		self.significant_func_list                         = significant_func_list
		self.linear_pattern_vec                            = linear_pattern_vec
		self.best_high_var_pattern_tagged_nodes            = best_high_var_pattern_tagged_nodes
		self.best_low_var_pattern_tagged_nodes             = best_low_var_pattern_tagged_nodes
		self.best_cross_covar_exposer_pattern_tagged_nodes = best_cross_covar_exposer_pattern_tagged_nodes

def set_global_variance_analysis_cet_results(var_anal_cet_results):
	global significant_func_list, linear_pattern_vec, \
			best_high_var_pattern_tagged_nodes, best_low_var_pattern_tagged_nodes, \
			best_cross_covar_exposer_pattern_tagged_nodes

	significant_func_list                         = var_anal_cet_results.significant_func_list
	linear_pattern_vec                            = var_anal_cet_results.linear_pattern_vec
	best_high_var_pattern_tagged_nodes            = var_anal_cet_results.best_high_var_pattern_tagged_nodes
	best_low_var_pattern_tagged_nodes             = var_anal_cet_results.best_low_var_pattern_tagged_nodes
	best_cross_covar_exposer_pattern_tagged_nodes = var_anal_cet_results.best_cross_covar_exposer_pattern_tagged_nodes

def del_global_variance_analysis_cet_results():
	global significant_func_list, linear_pattern_vec, \
			best_high_var_pattern_tagged_nodes, best_low_var_pattern_tagged_nodes, \
			best_cross_covar_exposer_pattern_tagged_nodes

	del significant_func_list, linear_pattern_vec, \
			best_high_var_pattern_tagged_nodes, best_low_var_pattern_tagged_nodes, \
			best_cross_covar_exposer_pattern_tagged_nodes

def cmp_variance_impact_metric(pat1, pat2):
	'''sort patterns descending'''
	vim1 = pat1.pred_behavior["variance_impact_metric"]
	vim2 = pat2.pred_behavior["variance_impact_metric"]

	if vim1 < vim2:
		return 1
	elif vim1 == vim2:
		return 0
	else:
		return -1



def get_pattern_indices_for_user_scope_id(linear_pattern_vec, user_scope_id):
	pattern_indices = []
	for i, pat in enumerate(linear_pattern_vec):
		if pat.user_scope_id == user_scope_id:
			pattern_indices.append(i)
	return pattern_indices

pattern_set_variance_impact_metric_factor = 0.10
def extract_pattern_set(vim_index_tuple_list):
	'''vim_index_tuple_list: list of (vim, index), where 'vim' is the variance_impact_metric of a pattern and
	    'index' is the index of the pattern in linear_pattern_vec
	Returns a sorted trimmed copy of vim_index_tuple_list
	''' 
	max_vim = 0.0
	for vim, index in vim_index_tuple_list:
		if vim > max_vim:
			max_vim = vim

	pattern_set = []
	for vim, index in vim_index_tuple_list:
		if vim >= pattern_set_variance_impact_metric_factor * max_vim:
			pattern_set.append( (vim, index) )	

	pattern_set.sort(lambda x, y: cmp(y[0], x[0]))
	return pattern_set



enable_binning_analysis = True
enable_extraction_of_unscoped_high_variance_patterns = True
enable_extraction_of_unscoped_low_variance_patterns = True

user_specified_scopes_for_high_variance_analysis = []
	#list of scopes; each scope is a list of form [(FN, LN), ... (F2, L2), (F1, L1)]
	#  where FN .. F1 must be present on stack in the given order (but not necessarily contiguously)
	#  for the given scope to "occur". F1 is outermost.
	# LN .. L1 give the corresponding func_lexical_ids that must match on the call-stack for the scope
	#    to occur. Li = None implies a "don't care" for matching the func_lexical_id of Fi.
def analyze_cet_and_find_patterns(given_profile_repr):
	set_global_profile_representation(given_profile_repr)

	### Variance Analysis

	common.MARK_PHASE_START("VARIANCE ANALYSIS")

	global significant_func_list
	significant_func_list = ccanalysis.get_sorted_significant_func_info_list(func_id_to_info, min_significant_exec_time)

	print "##### func_id_to_info = ", func_id_to_info
	print "##### significant_func_list = ", significant_func_list



	#### Classify func_nodes

	tag_variant_nodes_recursive(profile_cet)
	ccanalysis.determine_bounded_recursive(profile_cet, map_func_name_to_loop_hier, min_significant_exec_time)

	high_variant_distinguishing_context_map = {} # map from func_info to context
	low_variant_distinguishing_context_map = {} # map from func_info to context
	cross_covar_exposer_distinguishing_context_map = {} # map from func_info to context

	user_spec_high_var_scope_match_criteria_func_list = []
	for i, user_scope in enumerate(user_specified_scopes_for_high_variance_analysis):
		func_object = user_spec_high_var_scope_match_criteria_func_class(user_scope, ('High', i))
		user_spec_high_var_scope_match_criteria_func_list.append(func_object.match_criteria_func)

	user_spec_high_var_scopes_distinguishing_context_map_list = [{} for hv_scope in user_specified_scopes_for_high_variance_analysis]
		#list of maps from func_info to context.
		# Each list element corresponds to an entry in user_spec_high_var_scopes_distinguishing_context_map_list

	for f in significant_func_list:
		high_variant_distinguishing_context_map[f] = []
		if enable_extraction_of_unscoped_high_variance_patterns:
			high_variant_distinguishing_context_map[f] = \
				ccanalysis.extract_min_distinguishing_context_for_function_given_match_criteria(f, high_variant_match_criteria_func)

		low_variant_distinguishing_context_map[f] = []
		if enable_extraction_of_unscoped_low_variance_patterns:
			low_variant_distinguishing_context_map[f] = \
				ccanalysis.extract_min_distinguishing_context_for_function_given_match_criteria(f, low_variant_match_criteria_func)

		cross_covar_exposer_distinguishing_context_map[f] = \
			ccanalysis.extract_min_distinguishing_context_for_function_given_match_criteria(f, cross_covar_exposer_match_criteria_func)

		for i in range(len(user_specified_scopes_for_high_variance_analysis)):
			user_spec_high_var_scopes_distinguishing_context_map_list[i][f] \
				= ccanalysis.extract_min_distinguishing_context_for_function_given_match_criteria(f, user_spec_high_var_scope_match_criteria_func_list[i])

	#print "##### high_variant_distinguishing_context_map = ", high_variant_distinguishing_context_map
	#print "##### low_variant_distinguishing_context_map = ", low_variant_distinguishing_context_map
	#print "##### cross_covar_exposer_distinguishing_context_map = ", cross_covar_exposer_distinguishing_context_map

	common.MARK_PHASE_COMPLETION("VARIANCE ANALYSIS")

	### Do Binning: Profiling and Analysis
	if enable_binning_analysis:
		high_var_pattern_tagged_nodes = []
		for f in high_variant_distinguishing_context_map:
			high_var_pattern_tagged_nodes.extend( [m for m, cc in high_variant_distinguishing_context_map[f]] )

		cross_covar_exposer_pattern_tagged_nodes = []
		for f in cross_covar_exposer_distinguishing_context_map:
			cross_covar_exposer_pattern_tagged_nodes.extend( [m for m, cc in cross_covar_exposer_distinguishing_context_map[f]] )
		
		do_binning_profiling_and_analysis(high_var_pattern_tagged_nodes, cross_covar_exposer_pattern_tagged_nodes, given_profile_repr)


	### Further differentiate between patterns using Pattern Similarity Trees

	common.MARK_PHASE_START("PATTERN SIMILARITY TREE DIFFERENTIATION")

	ccanalysis.detach_and_differentiate_using_pst_groups(high_variant_distinguishing_context_map, high_variant_sm_func)
	ccanalysis.detach_and_differentiate_using_pst_groups(low_variant_distinguishing_context_map, low_variant_sm_func)
	ccanalysis.detach_and_differentiate_using_pst_groups(cross_covar_exposer_distinguishing_context_map, cross_covar_exposer_sm_func)

	for user_spec_scope_hv_dist_context_map in user_spec_high_var_scopes_distinguishing_context_map_list:
		ccanalysis.detach_and_differentiate_using_pst_groups(user_spec_scope_hv_dist_context_map, high_variant_sm_func)

	print "@@@@@@@@@@@@@@@@ After PST differentation: high_variant_distinguishing_context_map =", high_variant_distinguishing_context_map
	print "@@@@@@@@@@@@@@@@ After PST differentation: low_variant_distinguishing_context_map =", low_variant_distinguishing_context_map
	print "@@@@@@@@@@@@@@@@ After PST differentation: cross_covar_exposer_distinguishing_context_map =", cross_covar_exposer_distinguishing_context_map
	print "@@@@@@@@@@@@@@@@ After PST differentation: <user_spec_scope, user_spec_high_var_scopes_distinguishing_context_map_list> =", \
			zip(user_specified_scopes_for_high_variance_analysis, user_spec_high_var_scopes_distinguishing_context_map_list)

	common.MARK_PHASE_COMPLETION("PATTERN SIMILARITY TREE DIFFERENTIATION")
	#sys.exit()

	### Dump graph
	print "------------- DUMPING GRAPH ", "-------------"

	global best_high_var_pattern_tagged_nodes
	best_high_var_pattern_tagged_nodes = []
	for f in high_variant_distinguishing_context_map:
		best_high_var_pattern_tagged_nodes.extend( [m for m, cc in high_variant_distinguishing_context_map[f]] )
	clamp_at_top_fraction_of_func_nodes(best_high_var_pattern_tagged_nodes, 0.01) #GENERALIZE: Top-clamping is not the best criteria


	global best_low_var_pattern_tagged_nodes
	best_low_var_pattern_tagged_nodes = []
	for f in low_variant_distinguishing_context_map:
		best_low_var_pattern_tagged_nodes.extend( [m for m, cc in low_variant_distinguishing_context_map[f]] )

	global best_cross_covar_exposer_pattern_tagged_nodes
	best_cross_covar_exposer_pattern_tagged_nodes = []
	for f in cross_covar_exposer_distinguishing_context_map:
		best_cross_covar_exposer_pattern_tagged_nodes.extend( [m for m, cc in cross_covar_exposer_distinguishing_context_map[f]] )

	profile_cet.dump_graphviz_view("cet_graph_view.dot", significant_subtree_cutoff_test_func, node_decorator_func)
	#sys.exit(0)


	### Pattern Formation
	global linear_pattern_vec
	linear_pattern_vec = []

	num_high_var_patterns = linearpattern.create_detection_patterns(linear_pattern_vec, None, \
			high_variant_distinguishing_context_map, high_variant_setup_new_pattern_stats, \
			high_variant_combine_stats_with_pattern, significant_func_list, map_func_name_to_id, func_id_to_info)

	print "num_high_var_patterns =", num_high_var_patterns


	num_low_var_patterns = linearpattern.create_detection_patterns(linear_pattern_vec, None, \
			low_variant_distinguishing_context_map, low_variant_setup_new_pattern_stats, \
			low_variant_combine_stats_with_pattern, significant_func_list, map_func_name_to_id, func_id_to_info)

	print "num_low_var_patterns =", num_low_var_patterns

	num_cross_covar_patterns = linearpattern.create_detection_patterns(linear_pattern_vec, None, \
			cross_covar_exposer_distinguishing_context_map, cross_covar_exposer_setup_new_pattern_stats, \
			cross_covar_exposer_combine_stats_with_pattern, significant_func_list, map_func_name_to_id, func_id_to_info)

	print "num_cross_covar_patterns =", num_cross_covar_patterns

	num_user_spec_scope_high_var_patterns_list = [0] * len(user_spec_high_var_scopes_distinguishing_context_map_list)
	for i, user_spec_scope_hv_dist_context_map in enumerate(user_spec_high_var_scopes_distinguishing_context_map_list):
		num_user_spec_scope_high_var_patterns_list[i] = linearpattern.create_detection_patterns(linear_pattern_vec, ('High', i), \
				user_spec_scope_hv_dist_context_map, high_variant_setup_new_pattern_stats, \
				high_variant_combine_stats_with_pattern, significant_func_list, map_func_name_to_id, func_id_to_info)
	
	print "num_user_spec_scope_high_var_patterns_list =", num_user_spec_scope_high_var_patterns_list


	# Sort Linear Patterns based on variance_impact_metric
	for pat in linear_pattern_vec:
		if pat.pred_behavior.has_key("sq_err_sum"):
			std = math.sqrt( float(pat.pred_behavior["sq_err_sum"]) / float(pat.invoke_count_vec[0]) )
			variance_impact_metric = k * std * pat.invoke_count_vec[0]
		else:
			variance_impact_metric = 0.0
		pat.pred_behavior["variance_impact_metric"] = variance_impact_metric
	linear_pattern_vec.sort(cmp_variance_impact_metric)

	print "@@@ sorted Linear Patterns = "
	print "["
	for i in range(len(linear_pattern_vec)):
		print ("([%s]" % i), linear_pattern_vec[i]
		print "),"
	print "]"


	# Pattern Set: indices of only those patterns whose variance_impact_metric >= 10% of the max variance_impact_metric of the scope
	
	user_spec_scope_pattern_indices = []
	for scope_index in range(len(user_specified_scopes_for_high_variance_analysis)):
		user_spec_scope_pattern_indices.append(get_pattern_indices_for_user_scope_id(linear_pattern_vec, ('High', scope_index)))

	user_spec_scope_pattern_sets = []
	for scope_index in range(len(user_specified_scopes_for_high_variance_analysis)):
		vim_index_tuple_list = [ (linear_pattern_vec[pat_index].pred_behavior["variance_impact_metric"], pat_index) \
									for pat_index in user_spec_scope_pattern_indices[scope_index] ]
		user_spec_scope_pattern_sets.append( extract_pattern_set(vim_index_tuple_list) )
	
	print "@@@ user_spec_scope_pattern_indices =", user_spec_scope_pattern_indices
	print "@@@ user_spec_scope_pattern_sets =", user_spec_scope_pattern_sets
		
		
	print "DONE\n"

	sys.stdout.flush()


	# Cleanup
	del_global_profile_representation()

	var_anal_cet_results = VarianceAnalysisCETResults(significant_func_list, linear_pattern_vec, \
								best_high_var_pattern_tagged_nodes, best_low_var_pattern_tagged_nodes, \
								best_cross_covar_exposer_pattern_tagged_nodes)
	del_global_variance_analysis_cet_results()

	return var_anal_cet_results


#analyze_cet_and_find_patterns()

###################################################
###### INTERACTIVELY APPLY WHAT-IF SCENARIOS ######
###################################################


def do_interactive_mode(given_profile_repr):
	print
	print "-------- ENTERING INTERACTIVE MODE --------"
	print

	print "Enter one of following:"
	print "  <node-num>: <stat> = <new-value>"
	print "       # Find node-num from graph, stat = i/t/s for invoke_count/total_count/sq_err_sum"
	print
	print "  apply"
	print "       # Apply previous changes to nodes, re-run analysis, produce new patterns and update graph"
	print
	print "  regression"
	print "       # Proceed with regression analysis on current patterns"
	print
	print "  quit"

	import sys
	import re

	node_update_re = re.compile('(\d+)\s*:\s*([i|t|s])\s*=\s*([0-9.]+)')
	apply_re = re.compile('apply')
	quit_re = re.compile('quit')
	regression_re = re.compile('regression')

	var_anal_cet_results = None

	while True:
		user_input = raw_input("cet> ")

		if quit_re.match(user_input):
			print "Quitting"
			sys.exit(0)
		
		elif apply_re.match(user_input):
			print "Applying"
			var_anal_cet_results = analyze_cet_and_find_patterns(given_profile_repr)
			continue

		elif regression_re.match(user_input):
			print "Doing Regression"
			break

		#Neither quit or apply
		node_groups = node_update_re.match(user_input)
		if not node_groups:
			print "Invalid input. Please try again..."
			continue

		#matched node-update format
		node_index, update_stat, new_value = node_groups.group(1,2,3)
		node_index = int(node_index)
		print "Updating node %s , stat = %s, value = %s" % (node_index, update_stat, new_value)
		func_node = profile_cet.find_node_with_index(node_index)
		if func_node == None:
			print "Node with index %s not present in graph. Please try again..."
			continue
		if update_stat == 'i':
			func_node.invoke_count = int(new_value)
		elif update_stat == 't':
			func_node.total_count = int(new_value)
		else:
			func_node.sq_err_sum = float(new_value)
		continue

	return var_anal_cet_results

#if enable_interactive_mode:
#	do_interactive_mode()



################################
###### PERFORM REGRESSION ######
################################

import regression

regression_file_name = profile_file_name
regression_test_high_var_bins_prediction_schemes = True
def collect_regression_stats(linear_pattern_vec, func_id_to_name, map_called_NULL_FUNC_artificial_lexical_id):
	global pat_regression_info, regression_max_count
	pat_regression_info, regression_max_count \
		= regression.run_regression(linear_pattern_vec, func_id_to_name, regression_file_name, \
						map_called_NULL_FUNC_artificial_lexical_id, regression_test_high_var_bins_prediction_schemes)

	print "----- Detection QUALITY RESULTS -----"

	user_spec_scope_pattern_indices = []
	for scope_index in range(len(user_specified_scopes_for_high_variance_analysis)):
		user_spec_scope_pattern_indices.append(get_pattern_indices_for_user_scope_id(linear_pattern_vec, ('High', scope_index)))

	user_spec_scope_regression_pattern_sets = []
	for scope_index in range(len(user_specified_scopes_for_high_variance_analysis)): # for each user-spec-scope
		vim_index_tuple_list = []
		for pat_index in user_spec_scope_pattern_indices[scope_index]:
			reg_info = pat_regression_info[pat_index]
			if reg_info.invoke_count > 0:
				reg_std = math.sqrt( float(reg_info.sq_err_sum) / float(reg_info.invoke_count) )
			else:
				reg_std = 0.0
			variance_impact_metric = k * reg_std * reg_info.invoke_count
			reg_info.pred_scheme_res["variance_impact_metric"] = variance_impact_metric
			vim_index_tuple_list.append( (variance_impact_metric, pat_index) )

		user_spec_scope_regression_pattern_sets.append( extract_pattern_set(vim_index_tuple_list) )


	print "##### PATTERN MATCHING RESULTS:"
	print "["
	for i in range(len(linear_pattern_vec)):
		print ("([%s]" % i), linear_pattern_vec[i], ",", pat_regression_info[i]
		print "),"
	print "]"

	print "@@@ user_spec_scope_regression_pattern_sets =", user_spec_scope_regression_pattern_sets


	regression_epsilon = 0.10

	global num_low_var_pats, num_low_var_pats_that_remain_low_var, num_low_var_pats_with_similar_means
	global num_high_var_pats, num_high_var_pats_with_similar_std, num_high_var_pats_with_similar_means
	global num_cross_covar_pats

	num_low_var_pats = 0
	num_low_var_pats_that_remain_low_var = 0
	num_low_var_pats_with_similar_means = 0

	num_high_var_pats = 0
	num_high_var_pats_with_similar_std = 0
	num_high_var_pats_with_similar_means = 0

	num_cross_covar_pats = 0

	for pat, reg in zip(linear_pattern_vec, pat_regression_info):
		if pat.pred_behavior.has_key("cross_covar_exposer") and pat.pred_behavior["cross_covar_exposer"] == True:
			num_cross_covar_pats += 1
			continue

		pat_mean = pat.pred_behavior["mean"]
		if pat.pred_behavior["variant"] == "high":
			pat_std = pat.pred_behavior["std"]
			pat_cov = pat.pred_behavior["cov"]

		if reg.invoke_count == 0:
			(reg_mean, reg_std, reg_cov) = (0.0, 0.0, 0.0)
		else:
			reg_mean = float(reg.total_count) / float(reg.invoke_count)
			reg_std = math.sqrt( reg.sq_err_sum / float(reg.invoke_count) )
			reg_cov = reg_std / reg_mean

		similar_mean = (abs(pat_mean - reg_mean) < regression_epsilon * pat_mean)

		if pat.pred_behavior["variant"] == "high":
			num_high_var_pats += 1
			similar_std  = (abs(pat_std  - reg_std)  < regression_epsilon * pat_std)
			if similar_std:
				num_high_var_pats_with_similar_std += 1
			if similar_mean:
				num_high_var_pats_with_similar_means += 1

		if pat.pred_behavior["variant"] == "low":
			num_low_var_pats += 1
			if reg_cov < cov_low_threshold:
				num_low_var_pats_that_remain_low_var += 1
			if similar_mean:
				num_low_var_pats_with_similar_means += 1

#collect_regression_stats()



#################################################
########   PRINT FINAL QUALITY RESULTS   ########
#################################################

def update_early_pred_histogram(early_pred_histogram, pred_accuracy_required, pat_pred_probability):
		rev_pred_prob = pat_pred_probability[:]
		rev_pred_prob.reverse()
		check_flag = False
		for i, p in enumerate(rev_pred_prob):
			early_index = len(rev_pred_prob) - 1 - i
			if p >= pred_accuracy_required:
				if len(early_pred_histogram) <= early_index:
					early_pred_histogram.extend([0] * (early_index + 1 - len(early_pred_histogram)))
				early_pred_histogram[early_index] += 1
				check_flag = True
				break

		if not check_flag:
			sys.exit("update_early_pred_histogram(): ERROR: BUG in code")

def generate_pattern_set_statistics(pattern_list):
	length_histogram = []
	bounded_on_length_histogram = []

	early95_pred_histogram = []
	early90_pred_histogram = []

	max_exec_time_in_pattern_set = 0

	for pat in pattern_list:
		pat_len = len(pat.call_chain)
		if len(length_histogram) <= pat_len:
			length_histogram.extend([0] * (pat_len + 1 - len(length_histogram)))
			bounded_on_length_histogram.extend([0] * (pat_len + 1 - len(bounded_on_length_histogram)))
		length_histogram[pat_len] += 1
		if pat.pred_behavior["bounded"]:
			bounded_on_length_histogram[pat_len] += 1

		update_early_pred_histogram(early95_pred_histogram, 0.95, pat.pred_probability)
		update_early_pred_histogram(early90_pred_histogram, 0.90, pat.pred_probability)

		if pat.pred_behavior.has_key("variant"):
			max_exec_time_in_pattern_set = max(max_exec_time_in_pattern_set, \
											pat.invoke_count_vec[0] * pat.pred_behavior["mean"])

	fraction_max_exec_time_in_pattern_set = max_exec_time_in_pattern_set / float(max_count)
	return (length_histogram, bounded_on_length_histogram, early95_pred_histogram, early90_pred_histogram, fraction_max_exec_time_in_pattern_set)


def generate_high_var_binned_accuracy_statistics(zipped_high_var_binned_pat_and_regr_info):
	scheme_predict_last_percentiles = [0] * 10 # ranges: 0%-9%, 10%-11%, ..., 90% - 100%
	scheme_predict_2state_percentiles= [0] * 10 # ranges: 0%-9%, 10%-11%, ..., 90% - 100%
	
	for pat, regr_info in zipped_high_var_binned_pat_and_regr_info:
		pred_res_last = regr_info.pred_scheme_res["scheme_predict_last"]
		accuracy_last = pred_res_last["num_correct_preds"] / float(pred_res_last["count"])
		index_last = int(accuracy_last * 10)
		scheme_predict_last_percentiles[index_last] += 1

		pred_res_2state = regr_info.pred_scheme_res["scheme_predict_2state"]
		accuracy_2state = pred_res_2state["num_correct_preds"] / float(pred_res_2state["count"])
		index_2state = int(accuracy_2state * 10)
		scheme_predict_2state_percentiles[index_2state] += 1

	return (scheme_predict_last_percentiles, scheme_predict_2state_percentiles)
		
		


def generate_and_print_final_quality_results(given_profile_repr, var_anal_cet_results):
	set_global_profile_representation(given_profile_repr)
	set_global_variance_analysis_cet_results(var_anal_cet_results)

	print "@@@ Profiling: maximum instruction count = ", max_count

	print "@@@ Regression: maximum instruction count = ", regression_max_count

	print "@@@ Low-Variance-Pattern-Statistics @@@"
	print generate_pattern_set_statistics([pat for pat in linear_pattern_vec \
			if pat.pred_behavior.has_key("variant") and pat.pred_behavior["variant"] == "low"])

	print "@@@ High-Variance-Binned-Pattern-Statistics @@@"
	print generate_pattern_set_statistics([pat for pat in linear_pattern_vec \
			if pat.pred_behavior.has_key("high_var_bins")])

	print "@@@ Cross-CoVar-Binned-Pattern-Statistics @@@"
	print generate_pattern_set_statistics([pat for pat in linear_pattern_vec \
			if pat.pred_behavior.has_key("cross_covar_bins")])

	print "@@@ High-Variance-Unbinnable-Pattern-Statistics @@@"
	print generate_pattern_set_statistics([pat for pat in linear_pattern_vec \
			if pat.pred_behavior.has_key("variant") and pat.pred_behavior["variant"] == "high" \
				and not pat.pred_behavior.has_key("high_var_bins")])

	print "@@@ Cross-CoVar-Unbinnable-Pattern-Statistics @@@"
	print generate_pattern_set_statistics([pat for pat in linear_pattern_vec \
			if pat.pred_behavior.has_key("cross_covar_exposer") \
				and not pat.pred_behavior.has_key("cross_covar_bins")])


	print "@@@ Accuracy results for High-Variance-Binned-Patterns @@@"
	print generate_high_var_binned_accuracy_statistics( \
			[(pat, regr_info) for (pat, regr_info) in zip(linear_pattern_vec, pat_regression_info) \
									if pat.pred_behavior.has_key("high_var_bins")])

	print "num_low_var_pats =", num_low_var_pats
	print "num_low_var_pats_that_remain_low_var =", num_low_var_pats_that_remain_low_var
	print "num_low_var_pats_with_similar_means =", num_low_var_pats_with_similar_means
	print

	print "num_high_var_pats =", num_high_var_pats
	print "num_high_var_pats_with_similar_std =", num_high_var_pats_with_similar_std
	print "num_high_var_pats_with_similar_means =", num_high_var_pats_with_similar_means
	print

	print "num_cross_covar_pats =", num_cross_covar_pats

	del_global_variance_analysis_cet_results()
	del_global_profile_representation()

#generate_and_print_final_quality_results()


##################################################

def read_config_parameter_file(varanalysis_config_file_name = "config_varanalysis.py"):
	'''Reads 'config_varanalysis.py' in the execution directory, if it exists.
		The config file can override default values of global parameters by modifying
		them in python syntax'''

	### Modifiable globals
	# config_command is read-only
	global escape_bounds_probability
	global window_bound_high_var
	global window_bound_low_var
	global cov_high_threshold
	global cov_low_threshold

	global covar_fraction_for_high
	global covar_fraction_for_medium
	global covar_fraction_for_low
	global covar_exposer_factor

	global min_invoke_count_needed_for_pattern

	global high_variant_mean_epsilon
	global high_variant_cov_epsilon

	global low_variant_mean_epsilon

	global profile_file_name

	global regression_file_name

	global enable_cross_covar_analysis
	global enable_binning_analysis
	global enable_extraction_of_unscoped_high_variance_patterns
	global enable_extraction_of_unscoped_low_variance_patterns

	global run_regression_pass
	global regression_test_high_var_bins_prediction_schemes

	global user_specified_scopes_for_high_variance_analysis


	try:
		cf = open(varanalysis_config_file_name)
	except IOError:
		config_file_exists = False
	else:
		config_file_exists = True

	if(config_file_exists):
		cf.close()
		print "@@@ Applying configuration from config-file: ", varanalysis_config_file_name
		execfile(varanalysis_config_file_name, globals())
			# Update any of the the global parameters listed in 'display_varanalysis_config_parameters()' below
	

def display_varanalysis_config_parameters():
	print " ------------- varanalysis config parameters -------------"

	print "@Config: config_command =", config_command
	print "@Parameter: escape_bounds_probability =", escape_bounds_probability
	print "@Parameter: window_bound_high_var =", window_bound_high_var
	print "@Parameter: window_bound_low_var =", window_bound_low_var
	print "@Implied Parameter: cov_high_threshold =", cov_high_threshold
	print "@Implied Parameter: cov_low_threshold =", cov_low_threshold
	print
	print "@Parameter: covar_fraction_for_high =", covar_fraction_for_high
	print "@Parameter: covar_fraction_for_medium =", covar_fraction_for_medium
	print "@Parameter: covar_fraction_for_low =", covar_fraction_for_low
	print "@Parameter: covar_exposer_factor =", covar_exposer_factor
	print
	print "@Parameter: min_invoke_count_needed_for_pattern =", min_invoke_count_needed_for_pattern
	print
	print "@Parameter: high_variant_mean_epsilon =", high_variant_mean_epsilon
	print "@Parameter: high_variant_cov_epsilon =", high_variant_cov_epsilon
	print
	print "@Parameter: low_variant_mean_epsilon =", low_variant_mean_epsilon
	print
	print "@Parameter: profile_file_name =", profile_file_name.__repr__()
	print
	print "@Parameter: regression_file_name =", regression_file_name.__repr__()
	print
	print "@Parameter: enable_cross_covar_analysis =", enable_cross_covar_analysis
	print "@Parameter: enable_binning_analysis =", enable_binning_analysis
	print "@Parameter: enable_extraction_of_unscoped_high_variance_patterns =", enable_extraction_of_unscoped_high_variance_patterns
	print "@Parameter: enable_extraction_of_unscoped_low_variance_patterns =", enable_extraction_of_unscoped_low_variance_patterns
	print
	print "@Parameter: run_regression_pass =", run_regression_pass 
	print "@Parameter: regression_test_high_var_bins_prediction_schemes =", regression_test_high_var_bins_prediction_schemes
	print
	print "@Parameter: user_specified_scopes_for_high_variance_analysis =", user_specified_scopes_for_high_variance_analysis.__repr__()
	print
		
config_command = None
	#meant to be read by varanalysis-configuration in order to select configuration based on command-line flag

run_regression_pass = True
def run_varanalysis():
	### Process Command Line Flags
	global config_command

	enable_interactive_mode = False
	config_command_index = 1
	if len(sys.argv) >= 2 and sys.argv[1] == "-i":
		enable_interactive_mode = True
		config_command_index += 1
	
	if config_command_index < len(sys.argv):
		config_command = sys.argv[config_command_index]

	read_config_parameter_file()
	display_varanalysis_config_parameters()

	### Start
	new_profile_repr = read_profile_representation()

	var_anal_cet_results = analyze_cet_and_find_patterns(new_profile_repr)

	if enable_interactive_mode:
		tmp = do_interactive_mode(new_profile_repr)
		if tmp != None: #analysis was interactively invoked atleast once
			var_anal_cet_results = tmp

	if run_regression_pass:
		collect_regression_stats(var_anal_cet_results.linear_pattern_vec, new_profile_repr.func_id_to_name, \
									new_profile_repr.map_called_NULL_FUNC_artificial_lexical_id)

		generate_and_print_final_quality_results(new_profile_repr, var_anal_cet_results)
			#FIXME: also pass regression result as parameters


if __name__ == "__main__": #invoked as script
	run_varanalysis()
else:
	print "~~ IMPORTED varanalysis ~~"
