#!/usr/bin/python

# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com


#FIXME
# 1) cross_sq_err_sum is currently not being computed.

import sys
import math

import common

###################################################
########   DEFINE REGRESSION TECHNIQUES    ########
###################################################

################ Regression ################

class CallStackEntry:
	def __init__(self, func_name, func_lexical_id):
		self.func_name = func_name
		self.func_lexical_id = func_lexical_id
		self.entry_count = None
		self.exit_count = None

	def __repr__(self):
		result = "(%s, %s %s->%s)" % (self.func_name, self.func_lexical_id, self.entry_count, self.exit_count)
		return result


class PatternRegressionInfo:
	def __init__(self, pattern_len = None):
		self.index = pattern_len
		                       # Indicates where in call-chain pattern the top-of-stack call is.
		                       #   index == pattern_len implies that no prefix of the pattern matches
		                       #     the current stack
		                       #   index == 0 implies that the call at the top of stack completes
		                       #     the matching of the pattern.
		                       #   index < 0 implies that the entire pattern is on the stack, with
		                       #     with additional child calls

		self.stack_match_index = None
		                       # indicates the location in the call-stack where the 'index' matched

		self.total_count = 0   # total time spent in lowest call of this pattern
		self.invoke_count = 0  # total number of times the lowest call was invoked in this pattern

		self.sq_err_sum = 0.0  # sum of squared-errors in execution count over
		                       # invocations of the lowest call in this pattern

		self.cross_sq_err_sum = [] # 2-d matrix, holds sums for cross-covariance
		                       # between children function calls of current node

		self.pred_scheme_res = {}  # Holds results for how well different prediction schemes did at
                               # predicting execution-time

	def __repr__(self):
		if self.invoke_count == 0:
			(mean, std, cov) = (0.0, 0.0, 0.0)
		else:
			mean = float(self.total_count) / float(self.invoke_count)
			std = math.sqrt( self.sq_err_sum / float(self.invoke_count) )
			cov = std / mean

		result = "(%s/%s sq=%s mean=%s std=%s cov=%s accuracy=%s)" % (self.total_count, self.invoke_count, self.sq_err_sum, mean, std, cov, self.pred_scheme_res)
		return result


pred_correctness_epsilon = 0.10
def predict_and_verify_high_var_binned_pattern(pat, regr_info, count_diff):
	bins = pat.pred_behavior["high_var_bins"]
	
	# scheme_predict_last
	if not regr_info.pred_scheme_res.has_key("scheme_predict_last"):
		regr_info.pred_scheme_res["scheme_predict_last"] = {'last_value' : None, 'num_correct_preds' : 0, 'count' : 0}
	else:
		scheme_predict_last = regr_info.pred_scheme_res["scheme_predict_last"]

		prediction = scheme_predict_last["last_value"]
		if abs(count_diff - prediction) < pred_correctness_epsilon * count_diff:
			scheme_predict_last["num_correct_preds"] += 1
		scheme_predict_last["count"] += 1

	scheme_predict_last = regr_info.pred_scheme_res["scheme_predict_last"]
	scheme_predict_last["last_value"] = count_diff

	# scheme_predict_2state
	if not regr_info.pred_scheme_res.has_key("scheme_predict_2state"):
		regr_info.pred_scheme_res["scheme_predict_2state"] = {'curr_bin_index' : None, 'num_consecutive_incorrect' : 2, 'num_correct_preds' : 0, 'count' : 0}
	else:
		scheme_predict_2state = regr_info.pred_scheme_res["scheme_predict_2state"]

		prediction = bins[ scheme_predict_2state["curr_bin_index"] ][0]
		if abs(count_diff - prediction) < pred_correctness_epsilon * count_diff:
			scheme_predict_2state["num_correct_preds"] += 1
		else:
			scheme_predict_2state["num_consecutive_incorrect"] += 1
		scheme_predict_2state["count"] += 1

	scheme_predict_2state = regr_info.pred_scheme_res["scheme_predict_2state"]
	if scheme_predict_2state["num_consecutive_incorrect"] == 2:
		scheme_predict_2state["num_consecutive_incorrect"] = 0
		closest_index = None
		min_diff = None
		for i, bin_val in enumerate(bins):
			bin_centroid = bin_val[0]
			bin_diff = abs(bin_centroid - count_diff)
			if min_diff == None or bin_diff < min_diff:
				min_diff = bin_diff
				closest_index = i
		scheme_predict_2state["curr_bin_index"] = closest_index
		

	#dominant-bin prediction



def examine_call_stack_at_entry():
	#print call_stack

	top_call = call_stack[-1]

	for i in range(len(linear_pattern_vec)):
		pat = linear_pattern_vec[i]
		regr_info = pat_regression_info[i]

		if regr_info.stack_match_index != None and regr_info.stack_match_index + 2 != len(call_stack):
			# a prefix of the pattern previously matched on stack, but not just below top of stack
			continue

		if regr_info.index > 0: #either pattern not matched, or only a strict prefix matched so far
			#attempt match, increment regr_info.index only if incremental match found

			if regr_info.index == len(pat.call_chain): # start point, no func_lexical_id context
				context_call = (top_call.func_name, None)
			else:
				context_call = (top_call.func_name, top_call.func_lexical_id)

			if pat.call_chain[regr_info.index - 1] == context_call: #pattern matched incrementally
				regr_info.index -= 1
				regr_info.stack_match_index = len(call_stack) - 1
				#print "MATCHED: found at index = ", regr_info.index, " for pattern = ", pat

				if pass_number == 1:
					if regr_info.index == 0: #full pattern matched
						regr_info.invoke_count += 1

				elif pass_number == 2:
					pass # for now, since we are not yet making predictions based on cross-co-variance


		#else: # pattern already fully matched on stack, now entering children calls at bottom of pattern



def examine_call_stack_at_exit():
	#print call_stack

	top_call = call_stack[-1]
	count_diff = top_call.exit_count - top_call.entry_count

	for i in range(len(linear_pattern_vec)):
		pat = linear_pattern_vec[i]
		regr_info = pat_regression_info[i]

		if regr_info.stack_match_index == None or regr_info.stack_match_index != len(call_stack) - 1:
			# either no prefix of pattern exists on stack,
			#  or the exiting call is a child of a prefix of the pattern but not part of the pattern
			if regr_info.stack_match_index > len(call_stack) - 1:
				sys.exit("examine_call_stack_at_exit(): INTERNAL ERROR: stack_match_index = %s, with call-stack = %s\n for %s" \
						% (regr_info.stack_match_index, call_stack, pat))
			continue
		
		# Now: regr_info.stack_match_index == len(call_stack) - 1

		if regr_info.index == 0: # full pattern is starting to exit
			if pass_number == 1:
				regr_info.total_count += count_diff

				regr_info.sq_err_sum += count_diff**2
					#Computational formula for variance: Var(X) = E(X**2) - mean**2

			elif pass_number == 2:
				#mean = float(regr_info.total_count) / float(regr_info.invoke_count)
				#regr_info.sq_err_sum += (count_diff - mean)**2

				if pat.pred_behavior.has_key("high_var_bins"):
					predict_and_verify_high_var_binned_pattern(pat, regr_info, count_diff)
					

		regr_info.index += 1
		regr_info.stack_match_index -= 1

		if regr_info.index == len(pat.call_chain): # pattern has fully exited
			regr_info.stack_match_index = None




def regression_entry_event(func_name, func_lexical_id, count):
	global call_stack

	call_entry = CallStackEntry(func_name, func_lexical_id)
	call_entry.entry_count = count

	call_stack.append(call_entry)

	#Invoke rules here
	examine_call_stack_at_entry()


def regression_exit_event(func_name, func_lexical_id, count):
	global call_stack

	if func_name == "NULL_FUNC":
		regression_preprocess_NULL_FUNC_on_exit(count)

	call_exit = call_stack[-1] # call at top of stack
	call_exit.exit_count = count

	if call_exit.func_name != func_name or call_exit.func_lexical_id != func_lexical_id:
		sys.exit("regression_exit_event(): ERROR: Cannot pop (%s, %s) from call-stack = %s\n Current lineno = %s" \
				% (func_name, func_lexical_id, call_stack, lineno))


	#Invoke rules here
	examine_call_stack_at_exit()

	call_stack.pop()

def regression_identifier_event(func_name, count):
	global call_stack

	if len(call_stack) > 0 and call_stack[-1].func_name == "NULL_FUNC": #parent call is NULL_FUNC
		if not map_called_NULL_FUNC_artificial_lexical_id.has_key(func_name):
			map_called_NULL_FUNC_artificial_lexical_id[func_name] = len(map_called_NULL_FUNC_artificial_lexical_id)
		artificial_func_lexical_id = map_called_NULL_FUNC_artificial_lexical_id[func_name]
		regression_entry_event(func_name, artificial_func_lexical_id, count) # forcing func_lexical_id = 0

def regression_preprocess_NULL_FUNC_on_exit(count):
	global call_stack

	if len(call_stack) > 0 and call_stack[-1].func_name == "NULL_FUNC": #At top of stack
		return # allow normal processing, as no function-pointer was invoked

	if len(call_stack) > 1 and call_stack[-2].func_name != "NULL_FUNC": #At one below top of stack
		sys.exit("ERROR: regression_preprocess_NULL_FUNC_on_exit() called but exiting node is not within a NULL_FUNC\n" \
					+ " call_stack = %s\n Current lineno = %s" % (call_stack, lineno))

	# Now, Special pre-processing required since function at top of stack was called via function-pointer
	top_call = call_stack[-1]
	regression_exit_event(top_call.func_name, top_call.func_lexical_id, count)



def process_regression_event(func_name, func_lexical_id, prof_event_type, count):
	if(prof_event_type == "entry"):
		regression_entry_event(func_name, func_lexical_id, count)
	elif(prof_event_type == "exit"):
		regression_exit_event(func_name, func_lexical_id, count)
	elif(prof_event_type == "identifier"):
		regression_identifier_event(func_name, count)
	else:
		sys.exit("ERROR: Regression-Profile event type = " + prof_event_type + " is not supported.\n" \
					+ " Current lineno = " + str(lineno))



def finish_variance_computation_step(reg_info):
	if reg_info.invoke_count > 0:
		mean = float(reg_info.total_count) / float(reg_info.invoke_count)
	else: #possible that some patterns are never invoked during regression
		mean = 0.0
	reg_info.sq_err_sum -= reg_info.invoke_count * (mean ** 2)

	# Due to numerical precision issues, and the two stage variance computation (subtraction),
	#   variance sums close to zero sometimes occur as small negative numbers
	#  => Zero clamping
	if reg_info.sq_err_sum < 0.0:
		reg_info.sq_err_sum = 0.0


def finish_variance_computation(pat_regression_info):
	for reg_info in pat_regression_info:
		finish_variance_computation_step(reg_info)


import re
profile_re = re.compile('(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)')

force_max_regression_steps =  60000000
regression_report_steps =     10000000
def regression_pass():
	global pass_number
	pass_number += 1

	global pat_regression_info

	common.MARK_PHASE_START("REGRESSION PASS # " + str(pass_number))
	profile_handle = open(profile_file_name, "r")

	process_regression_event("main", None, "entry", 0)

	global lineno
	lineno = 0
	regression_max_count = 0
	file_counter = 1
	continue_reading = True
	while continue_reading:
		for line in profile_handle:
			lineno += 1

			try:
				func_id, func_lexical_id, loop_lexical_id, prof_event_type, count \
						= profile_re.match(line).group(1,2,3,4,5)
			except:
				sys.exit("regression_pass(): syntax error in profile.dump at lineno = " + str(lineno) \
					+ " line = " + str(line))

			func_id = int(func_id)
			func_lexical_id = int(func_lexical_id)
			loop_lexical_id = int(loop_lexical_id)
			count = int(count)

			func_name = func_id_to_name[func_id]
			process_regression_event(func_name, func_lexical_id, prof_event_type, count)
			regression_max_count = count

			if lineno % regression_report_steps == 0:
				print "regression_pass(): Completed", lineno, "steps", "at time", common.get_curr_time()
				sys.stdout.flush()

			if force_max_regression_steps != None and lineno >= force_max_regression_steps:
				continue_reading = False
				break

		profile_handle.close()
		
		try:
			profile_handle = open(profile_file_name + "." + str(file_counter), "r")
			print "Opening file %d..." % (file_counter)
			file_counter += 1
		except IOError:
			break

	# Simulate exit up the call-chain (for incomplete profiles)
	while len(call_stack) > 0:
		top_call = call_stack[-1]
		process_regression_event(top_call.func_name, top_call.func_lexical_id, "exit", regression_max_count)

	common.MARK_PHASE_COMPLETION("REGRESSION PASS # " + str(pass_number))
	return regression_max_count


#####################################################
################ Regression User API ################
#####################################################


def run_regression(given_linear_pattern_vec, given_func_id_to_name, given_profile_file_name = "profile.dump", \
		given_map_called_NULL_FUNC_artificial_lexical_id = None, regression_test_high_var_bins_prediction_schemes = True):
	global linear_pattern_vec
	global func_id_to_name
	global pat_regression_info

	global profile_file_name
	profile_file_name = given_profile_file_name

	global map_called_NULL_FUNC_artificial_lexical_id
	map_called_NULL_FUNC_artificial_lexical_id = given_map_called_NULL_FUNC_artificial_lexical_id

	global pass_number
	pass_number = 0

	global call_stack
	call_stack = [] # stack with each element an instance of CallStackEntry

	linear_pattern_vec, func_id_to_name = given_linear_pattern_vec, given_func_id_to_name

	pat_regression_info = len(linear_pattern_vec) * [None]
	for i in range(len(pat_regression_info)):
		pat_regression_info[i] = PatternRegressionInfo( len(linear_pattern_vec[i].call_chain) )


	regression_max_count = regression_pass() #pass 1
	finish_variance_computation(pat_regression_info)
	if regression_test_high_var_bins_prediction_schemes:
		regression_pass() #pass 2

	return (pat_regression_info, regression_max_count)

