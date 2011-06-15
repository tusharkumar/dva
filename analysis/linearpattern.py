# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com


import common

class LinearPattern:
	def __init__(self):
		self.user_scope_id = None  # ID of the user-specified scope to which this pattern corresponds.
		                           #   =None if no scope specified

		self.call_chain = []       # list of (func_name, func_lexical_id) items.
		                           # First item is the top function in the call-stack for the call-chain represented, rest are
		                           #   its parents in order.
		                           # Last item has func_lexical_id = None, as it has no parent function whose func_lexical_id needs
		                           #   to be represented.

		self.invoke_count_vec = [] # Invocation counts for corresponding nodes in self.call_chain[] that produced this pattern

		self.exec_time_weight = 0  # Amount of execution time affected by prediction of this pattern

		self.pred_probability = [] # Of same length as self.call_chain. self.pred_probability[i] represents the probability that
		                           #  the suffix call-chain self.call_chain[0..i] will occur, conditioned on the occurence of the
		                           #  prefix call-chain self.call_chain[(i+1)..].
		                           #  Therefore, self.pred_probability[0] = 1.0

		self.pred_behavior = {}    # Map representing the behavior being predicted

		self.pattern_func_nodes = []
		                           # List of func_nodes at which this pattern was observed

	def __repr__(self):
		result = "\n"
		result += common.indent_string()
		result += "Pattern:\n"

		result += common.indent_string()
		result += "  user_scope_id = %s\n" % (self.user_scope_id, ) #special, since this is a tuple

		result += common.indent_string()
		result += "  invoke_count_vec = %s\n" % (self.invoke_count_vec)

		result += common.indent_string()
		result += "  call_chain = %s\n" % (self.call_chain)

		result += common.indent_string()
		result += "  pred_probability = %s\n" % (self.pred_probability)

		result += common.indent_string()
		result += "  exec_time_weight = %s\n" % (self.exec_time_weight)

		result += common.indent_string()
		result += "  pred_behavior = %s\n" % (self.pred_behavior)

		return result


def calculate_pred_probability(f_pat, map_func_name_to_id, func_id_to_info):
	for index in range(1, len(f_pat.call_chain)):
		prefix_call_chain = f_pat.call_chain[index:]

		indexed_func_name = prefix_call_chain[0][0]
		indexed_func_id = map_func_name_to_id[indexed_func_name]

		prefix_invoke_count = 0 # number of times this prefix was invoked in profile data
		for idx_func_node in func_id_to_info[indexed_func_id].significant_nodes:
			idx_func_call_chain = idx_func_node.get_call_chain_context()[0:len(prefix_call_chain)]
			idx_func_call_chain[-1] = (idx_func_call_chain[-1][0], None) # set base call's func_lexical_id to None

			if idx_func_call_chain == prefix_call_chain:
				prefix_invoke_count += idx_func_node.invoke_count
				# Note: This doesn't care whether the ignored suffix of f_pat's call-chain got invoked or not after idx_func_call_chain

		f_pat.pred_probability[index] = float(f_pat.invoke_count_vec[index]) / float(prefix_invoke_count)






import sys

def create_detection_patterns(linear_pattern_vec, user_scope_id, distinguishing_context_map, setup_new_pattern_stats, combine_stats_with_pattern, significant_func_list, map_func_name_to_id, func_id_to_info):
	'''Return number of patterns added to linear_pattern_vec
		Parms:
			linear_pattern_vec = list of instances of LinearPattern, to which new patterns are to be appended

			user_scope_id = identifies the user-specified-scope to which this pattern corresponds

			distinguishing_context_map = map of distinguishing-contexts indexed by FuncInfo instances,

			                             distinguishing-contexts = list of tuples of form
			                             (CET func_node, distinguishing-call-chain for func_node)
			                                where all of the func_nodes exhibit the same "specific type of behavior"
			                             and correspond to the function identified by the current FuncInfo instance

			setup_new_pattern_stats = a function that takes (func_node, LinearPattern instance) and sets up
			                                the 'exec_time_weight' and 'pred_behavior' fields of the
			                                LinearPattern instance, based on knowledge of the specific behavior

			combine_stats_with_pattern = a function that takes (CET func_node, existing instance of LinearPattern)
			                                as input, and returns whether (True/False) the given func_node can be
			                                adequately represented (based on knowledge of the current specific behavior)
											using the given instance of LinearPattern. If True, this function must
			                                also merge behavior specific attributes of func_node into the
											'exec_time_weight' and 'pred_behavior' statistics of the given
											instance of LinearPattern.
	'''

	orig_num_pats = len(linear_pattern_vec)

	for f in significant_func_list:
		f_lin_pats = []  # linear patterns found so far for function 'f', instances of LinearPattern

		for dist_context in distinguishing_context_map[f]:
			func_node = dist_context[0]
			call_chain = dist_context[1][:] # copy for following modification
			call_chain[-1] = (call_chain[-1][0], None) # Set func_lexical_id of highest parent to None

			if user_scope_id != None: # scoping matters
				# Now verify func_node conformance to user_scope_id
				if (not func_node.analysis.has_key("user_scope_id_list")) \
						or (user_scope_id not in func_node.analysis["user_scope_id_list"]):
					sys.exit("create_detection_patterns(): ERROR: user_scope_id = %s not found in func_node = %s\n" \
							 % (user_scope_id, func_node))

			matched_pat = None
			for pat in f_lin_pats:
				if pat.call_chain == call_chain and combine_stats_with_pattern(func_node, pat):
					# already have pattern, and statistics match up as well
					matched_pat = pat
					break

			if matched_pat == None:
				new_pat = LinearPattern()
				new_pat.user_scope_id = user_scope_id
				new_pat.call_chain = call_chain

				new_pat.invoke_count_vec = len(new_pat.call_chain) * [0]

				new_pat.pred_probability = len(new_pat.call_chain) * [0.0]
				new_pat.pred_probability[0] = 1.0

				new_pat.pred_behavior["bounded"] = func_node.analysis["bounded"]

				setup_new_pattern_stats(func_node, new_pat)

				curr_pat = new_pat

				f_lin_pats.append(new_pat)

			else:
				if not "COMBO" in matched_pat.pred_behavior:
					matched_pat.pred_behavior["COMBO"] = 1
				matched_pat.pred_behavior["COMBO"] += 1

				matched_pat.pred_behavior["bounded"] \
					= matched_pat.pred_behavior["bounded"] and func_node.analysis["bounded"]

				curr_pat = matched_pat

			cf = func_node
			for i in range(len(curr_pat.call_chain)):
				curr_pat.invoke_count_vec[i] += cf.invoke_count
				cf = cf.parent_func_node

			curr_pat.pattern_func_nodes.append(func_node)


		# Now, all patterns for 'f' have been found
		for f_pat in f_lin_pats:
			calculate_pred_probability(f_pat, map_func_name_to_id, func_id_to_info)

		linear_pattern_vec += f_lin_pats

	final_num_pats = len(linear_pattern_vec)

	return final_num_pats - orig_num_pats #number of patterns added

