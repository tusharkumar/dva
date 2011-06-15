# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com


import common

class PerInvokeDistribution:
	'''Captures the distribution of the number of times a given child entity was invoked per invocation
	    of its parent entity
	'''

	def __init__(self):
		self.curr_count = 0
		self.distr = []

	def update_stats_at_parent_exit(self):
		count = int(self.curr_count)

		pow_index = -1
		while count != 0:
			pow_index += 1
			if pow_index == len(self.distr):
				self.distr.append(0)

			count = count / 2

		if pow_index >= 0:
			self.distr[pow_index] += 1 #update stats

		self.curr_count = 0

	def __repr__(self):
		return self.distr.__repr__()



class FuncNode:
	"Represents a function call in the Call Execution Tree (CET)"

	def __init__(self, func_id, func_name, \
			parent_func_node = None, func_lexical_id = None, loop_lexical_id = None, \
			total_count = 0, invoke_count = 0, sq_err_sum = 0.0
	):
		self.func_id = func_id      # a unique number corresponding to the 'func_name'
		self.func_name = func_name

		self.parent_func_node = parent_func_node # must be an instance of FuncNode
		self.func_lexical_id = func_lexical_id
		                               # lexical-id of the current function call
		                               #   in the context of its parent

		self.loop_lexical_id = loop_lexical_id
		                               # lexical-id of innermost loop in parent function's
		                               # scope that contains the current function call


		self.funcs_called = [] # a list of instances of FuncNode or None,
		                       # indexed by the lexical-id of the child call
		                       #  in the scope of the current function.
		                       # None entries for calls not as yet executed


		self.total_count = total_count
		                       # the total time spent in this function-call across
		                       # all its invocations in the current context, including
							   # all the time spent in its children nodes

		self.invoke_count = invoke_count
		                       # the total number of times the parent made this
		                       # function call in the same context

		self.sq_err_sum = sq_err_sum # sum of squared-errors in execution count over
		                       # invocations of this function

		self.cross_sq_err_sum = [] # 2-d matrix, holds sums for cross-covariance
		                       # between children function calls of current node

		self.funcs_per_invoke_distr = [] # a list giving the invocation-count distribution
		                       # of each child function per invocation of the current func_node.
		                       # Elements correspond to elements of 'funcs_called'
							   # Each element is an instance of PerInvokeDistribution or
		                       #  is None (for a never invoked child).
		                       # FIXME: not dumped yet

		self.analysis = {}     # contains analysis results


	def __repr__(self):
		#global indent_depth

		result = common.indent_string()
		result += "FuncNode(func_id = %s, func_name = %s, func_lexical_id = %s, loop_lexical_id = %s, total_count = %s, invoke_count = %s, sq_err_sum = %s)\n" \
			% (self.func_id, self.func_name, self.func_lexical_id, self.loop_lexical_id, self.total_count, self.invoke_count, self.sq_err_sum)

		result += common.indent_string() + ".analysis = " + self.analysis.__repr__() + "\n" #GENERALIZE, CONTROL
		result += common.indent_string() + ".cross_sq_err_sum:\n"
		common.indent_depth += 1
		for i in range(len(self.cross_sq_err_sum)):
			result += common.indent_string()
			for j in range(len(self.cross_sq_err_sum)):
				result += self.cross_sq_err_sum[i][j].__repr__() + " "
			result += "\n"
		common.indent_depth -= 1

		result += common.indent_string() + ".funcs_called = [\n"
		common.indent_depth += 1
		for c in self.funcs_called:
			if c == None: #special case
				result += common.indent_string() + "Uninvoked-child" + "\n\n"
			else:
				result += c.__repr__()
		common.indent_depth -= 1
		result += common.indent_string() + "]\n"
		result += common.indent_string() + ".funcs_per_invoke_distr = " + self.funcs_per_invoke_distr.__repr__() + "\n"

		result += "\n"

		return result


	def get_call_chain_context(self):
		call_chain_context = [(self.func_name, self.func_lexical_id)]
		if self.parent_func_node != None:
			call_chain_context += self.parent_func_node.get_call_chain_context()

		return call_chain_context

	def get_call_chain_context_to_named_ancestor(self, ancestor_name):
		full_ccc = self.get_call_chain_context()
		anc_ccc = []
		found = False
		for (fn, lex_id) in full_ccc:
			if fn == ancestor_name:
				found = True
				break
			else:
				anc_ccc.append( (fn, lex_id) )
		if found == False:
			sys.exit("get_call_chain_context_to_named_ancestor(): ERROR: ancestor_name = %s is not an ancestor" \
						"for node = " % (ancestor_name, self.__repr__()) )
		return anc_ccc

	def depth_first_traverse(self, traverse_criteria):
		end_node = self

		mode = "forward" # forward/backward
		dft_node = self

		while 1:
			if mode == "backward" and dft_node == end_node:
				return # finished traversal

			if mode == "forward":
				if traverse_criteria(dft_node):
					yield dft_node
				else:
					mode = "backward"
					continue # skip sub-tree

				# Traverse to first child
				c = None
				for c in dft_node.funcs_called:
					if c != None:
						break

				if c == None: # no children
					mode = "backward"
				else:
					dft_node = c

			else: # mode == "backward"
				parent = dft_node.parent_func_node
				index = parent.funcs_called.index(dft_node) # dft_node's index

				# find next sibling
				c = None
				for c in range(index+1, len(parent.funcs_called)):
					if c != None:
						break

				if c == None: # no other sibling
					dft_node = parent
				else:
					mode = "forward"
					dft_node = c

	def dump_graphviz_view(self, graphfilename, significant_subtree_cutoff_test_func = None, node_decorator_func = None):
		"Annotate nodes with node_index and dump to file"
		fhandle = open(graphfilename, "w")
		fhandle.write('digraph G {\n')
		fhandle.write('  node [shape=record style=filled];\n')
		fhandle.write('  edge [color=grey75];\n')
		fhandle.write('  ranksep="1.0 equally";\n')

		node_count = [0] # to force the count to be "globally updated" within recursive_dump()

		def default_node_decorator_func(cn):
			return 'label="%s: %s"' % (cn.node_index, cn.func_name)

		if node_decorator_func == None:
			node_decorator_func = default_node_decorator_func

		def recursive_dump(cn, node_count, parent_node_index = None):
			node_count[0] += 1
			if significant_subtree_cutoff_test_func == None or significant_subtree_cutoff_test_func(cn):
				cn.node_index = node_count[0]

				fhandle.write( '%s [%s];\n' % (cn.node_index, node_decorator_func(cn)) )

				if parent_node_index != None:
					fhandle.write( '"%s" -> "%s";\n' % (parent_node_index, cn.node_index) )
				for i, c in enumerate(cn.funcs_called):
					if c != None:
						recursive_dump(c, node_count, cn.node_index)
					#else:
					#	fhandle.write( '"%s" -> 0;\n' % (cn.func_name) )

		recursive_dump(self, node_count)

		fhandle.write("}\n")
		fhandle.close()

	def clear_graph_data(self):
		if 'node_index' in dir(self):
			del self.node_index
			for c in self.funcs_called:
				if c != None:
					c.clear_graph_data()


	def find_node_with_index(self, node_index):
		"Return func_node with matching node_index in subtree, or None if not found"
		if 'node_index' in dir(self):
			if self.node_index == node_index:
				return self

			for c in self.funcs_called:
				if c != None:
					ret = c.find_node_with_index(node_index)
					if ret != None:
						return ret

		return None





class FuncInfo:
	"Represents a function (not on a per call-basis)"

	def __init__(self, func_name, exec_time = 0):
		self.func_name = func_name
		self.exec_time = exec_time

		self.significant_nodes = []
		    #list of instances of FuncNode in the CET that represent significant
		    #  execution time. All instances must be for the function called 'func_name'

	def __repr__(self):
		#global indent_depth

		result = common.indent_string()
		result += "FuncInfo(func_name = %s, exec_time = %s) " % (self.func_name, self.exec_time)
		return result




################ Reading func_map.dump ################

import re
func_map_re = re.compile('([A-Za-z0-9_.]+)\s+(\d+)')

def read_func_map(func_map_file_name = "func_map.dump"):

	func_id_to_name = []        # array: func_name indexed by func_id
	map_func_name_to_id = {}    # reverse lookup of func_id using func_name

	func_map_handle = open(func_map_file_name, "r")
	for line in func_map_handle:
		func_name, func_id = func_map_re.match(line).group(1,2)
		func_id = int(func_id)

		if func_id >= len(func_id_to_name):
			func_id_to_name.extend( (func_id - len(func_id_to_name) + 1) * [None] )
			# extend array size as necessary to accomodate loop
		
		func_id_to_name[func_id] = func_name
		map_func_name_to_id[func_name] = func_id

		print "func_name =", func_name, "func_id =", func_id

	func_map_handle.close()

	func_id_to_info = len(func_id_to_name) * [None]
		# array of instances of FuncInfo, indexed by corresponding func_ids

	for i in range(len(func_id_to_info)):
		func_id_to_info[i] = FuncInfo(func_id_to_name[i])

	return (func_id_to_name, map_func_name_to_id, func_id_to_info)

#func_id_to_name, map_func_name_to_id, func_id_to_info = read_func_map()


################ Reading func_loop_hier.dump ################

def read_func_loop_hier(func_loop_hier_file_name = "func_loop_hier.dump", need_special_NULL_FUNC = False):

	map_func_name_to_loop_hier = {}

	func_loop_hier_re = re.compile('([A-Za-z0-9_.]+)\s+=\s+(.*$)')

	func_loop_hier_handle = open(func_loop_hier_file_name, "r")
	for line in func_loop_hier_handle:
		func_name, loop_hier = func_loop_hier_re.match(line).group(1,2)

		command = "loop_hier = " + loop_hier
		exec(command) #coverts string to hierarchical data

		map_func_name_to_loop_hier[func_name] = loop_hier

		print "func_name =", func_name, "loop_hier =", loop_hier

	func_loop_hier_handle.close()

	if (need_special_NULL_FUNC == True) and (map_func_name_to_loop_hier.has_key("NULL_FUNC") == False):
		map_func_name_to_loop_hier["NULL_FUNC"] = (0, [])

	return map_func_name_to_loop_hier

#map_func_name_to_loop_hier = read_func_loop_hier(
#		need_special_NULL_FUNC = (map_func_name_to_id.has_key("NULL_FUNC") == True))

################ Reading profile.dump ################


import sys
def process_entry_event(func_id, func_lexical_id, loop_lexical_id, count):
	global curr_node

	func_name = func_id_to_name[func_id]

	extend_len = func_lexical_id - len(curr_node.funcs_called) + 1
	if extend_len > 0:
		curr_node.funcs_called.extend(extend_len * [None])
		curr_node.funcs_per_invoke_distr.extend(extend_len * [None])

	child = curr_node.funcs_called[func_lexical_id]
	if child == None:
		if pass_number != 1:
			sys.exit("ERROR: New entry event found on pass " + pass_number.__repr__() + \
					" at lineno = %s and call-context = %s" % (lineno, child.get_call_chain_context()))

		child = FuncNode(func_id, func_name, curr_node, func_lexical_id, loop_lexical_id)
		curr_node.funcs_called[func_lexical_id] = child

	else: # verify
		parms = (func_id, func_name, func_lexical_id, loop_lexical_id)
		if (child.func_id, child.func_name, child.func_lexical_id, child.loop_lexical_id) != parms:
			sys.exit("ERROR: entry parms = " + parms.__repr__() + \
					" do not match existing child node = " + child.__repr__() + "\n" + \
					" at lineno = %s and call-context = %s" % (lineno, child.get_call_chain_context()))

	child.last_entry_count = count

	if pass_number == 1:
		child.invoke_count += 1
		if curr_node.funcs_per_invoke_distr[func_lexical_id] == None:
			curr_node.funcs_per_invoke_distr[func_lexical_id] = PerInvokeDistribution()
		curr_node.funcs_per_invoke_distr[func_lexical_id].curr_count += 1

	elif pass_number >= 2:
		child.children_exec_counts = len(child.funcs_called) * [0]

	curr_node = child

	#print "after entry: curr_node = ", curr_node.func_name




def process_exit_event(func_id, func_lexical_id, loop_lexical_id, count):
	global curr_node
	global func_id_to_info

	func_name = func_id_to_name[func_id]

	if func_name == "NULL_FUNC":
		preprocess_NULL_FUNC_on_exit(count)

	parms = (func_id, func_name, func_lexical_id, loop_lexical_id)
	if (curr_node.func_id, curr_node.func_name, curr_node.func_lexical_id, curr_node.loop_lexical_id) \
			!= parms:
		sys.exit("ERROR: exit parms = " + parms.__repr__() + \
				" do not match curr_node = " + curr_node.__repr__() + \
				" at lineno = %s and call-context = %s" % (lineno, curr_node.get_call_chain_context()))

	count_diff = count - curr_node.last_entry_count
	del curr_node.last_entry_count

	if pass_number == 1:
		curr_node.total_count += count_diff
		func_id_to_info[func_id].exec_time += count_diff

		curr_node.sq_err_sum += count_diff ** 2
			#Computational formula for variance: Var(X) = E(X**2) - mean**2

		for cpid in curr_node.funcs_per_invoke_distr:
			if cpid != None:
				cpid.update_stats_at_parent_exit()

	elif pass_number == 2:
		if len(curr_node.cross_sq_err_sum) == 0:
			matrix_size = len(curr_node.funcs_called) + 1
			curr_node.cross_sq_err_sum = matrix_size * [None]
			for i in range(len(curr_node.cross_sq_err_sum)):
				curr_node.cross_sq_err_sum[i] = matrix_size * [0.0]


		# We want to also compare co-variance against the execution time of just the
		#  current function call (not including execution time spent in children calls)

		# the following data-structures will 'all_' prefix add the execution time of
		#  just the current call to the list

		#children_total_counts = [child.total_count for child in curr_node.funcs_called]
		children_total_counts = len(curr_node.funcs_called) * [0]
		for i, child in enumerate(curr_node.funcs_called):
			if child != None:
				children_total_counts[i] = child.total_count


		node_internal_total_count = curr_node.total_count - sum(children_total_counts)
		all_total_counts = children_total_counts + [node_internal_total_count]
		all_means = [float(total) / float(curr_node.invoke_count) for total in all_total_counts]

		node_internal_exec_count = count_diff - sum(curr_node.children_exec_counts)
		all_exec_counts = curr_node.children_exec_counts + [node_internal_exec_count]

		for i, (ci_exec_count, ci_mean) in enumerate( zip(all_exec_counts, all_means) ):
			for j, (cj_exec_count, cj_mean) in enumerate( zip(all_exec_counts, all_means) ):
				curr_node.cross_sq_err_sum[i][j] \
						+= (ci_exec_count - ci_mean) * (cj_exec_count - cj_mean)

		del curr_node.children_exec_counts

		if curr_node.parent_func_node != None: # check for "main" function node
			curr_index = curr_node.parent_func_node.funcs_called.index(curr_node)
			curr_node.parent_func_node.children_exec_counts[curr_index] += count_diff

	elif pass_number >= 3:
		user_post_order_visit_func(curr_node, count_diff, curr_node.children_exec_counts)

		del curr_node.children_exec_counts

		if curr_node.parent_func_node != None: # check for "main" function node
			curr_index = curr_node.parent_func_node.funcs_called.index(curr_node)
			curr_node.parent_func_node.children_exec_counts[curr_index] += count_diff


	curr_node = curr_node.parent_func_node

	
#map_called_NULL_FUNC_artificial_lexical_id = {}
# For each function-name called via a NULL_FUNC, assigns a unique ID that will
#  serve as the func_lexical_id of any instance of that function called from under any NULL_FUNC

def process_identifier_event(func_id, count):
	global curr_node

	if curr_node.func_name != "NULL_FUNC":
		return # not inside a function invoked from a function-pointer, nothing needs to be done

	# Now: curr_node.func_name == "NULL_FUNC"

#	#check if func_id already exists as one of the children of NULL_FUNC, else create new entry for it
#	found_loc = None
#	for i, c in enumerate(curr_node.funcs_called):
#		if c.func_id == func_id: # all entries would be defined, as they are allocated only when needed
#			found_loc = i
#	if found_loc == None:
#		found_loc = len(curr_node.funcs_called)
#
#	process_entry_event(func_id, found_loc, 0, count)

	called_func_name = func_id_to_name[func_id]
	if not map_called_NULL_FUNC_artificial_lexical_id.has_key(called_func_name):
		map_called_NULL_FUNC_artificial_lexical_id[called_func_name] = len(map_called_NULL_FUNC_artificial_lexical_id)
	artificial_func_lexical_id = map_called_NULL_FUNC_artificial_lexical_id[called_func_name]

	process_entry_event(func_id, artificial_func_lexical_id, 0, count)
	
	
def preprocess_NULL_FUNC_on_exit(count):
	global curr_node

	if curr_node.func_name == "NULL_FUNC":
		return # allow normal processing, as no function-pointer child was created

	if curr_node.parent_func_node.func_name != "NULL_FUNC":
		sys.exit("ERROR: preprocess_NULL_FUNC_on_exit() called but curr_node is not contained in NULL_FUNC parent\n" \
					+ "   curr_node.func_name = " + curr_node.func_name \
					+ " parent.func_name = " + curr_node.parent_func_node.func_name + "\n"
					+ " call-chain-context = " + curr_node.get_call_chain_context() + "\n"
					+ " at lineno = " + str(lineno))

	# Now, Special pre-processing required since function corresponding to curr_node was called via function-pointer
	process_exit_event(curr_node.func_id, curr_node.func_lexical_id, curr_node.loop_lexical_id, count)
	
	



def process_profile_event(func_id, func_lexical_id, loop_lexical_id, prof_event_type, count):
	if func_id < 0 or func_id >= len(func_id_to_name):
		sys.exit("ERROR: Unknown func_id when processing profile event at lineno=%s: func_id=%s func_lexical_id=%s loop_lexical_id=%s prof_event_type=%s count=%s" \
				% (lineno, func_id, func_lexical_id, loop_lexical_id, prof_event_type, count))
	func_name = func_id_to_name[func_id]
	if(prof_event_type == "entry"):
		process_entry_event(func_id, func_lexical_id, loop_lexical_id, count)
	elif(prof_event_type == "exit"):
		process_exit_event(func_id, func_lexical_id, loop_lexical_id, count)
	elif(prof_event_type == "identifier"):
		process_identifier_event(func_id, count)
	else:
		sys.exit("ERROR: Profile event type = " + prof_event_type + " is not supported."
					+ " lineno = " + str(lineno))


def finish_variance_computation_recursive(curr_node):
	mean = float(curr_node.total_count) / float(curr_node.invoke_count)
	curr_node.sq_err_sum -= curr_node.invoke_count * (mean ** 2)

	# Due to numerical precision issues, and the two stage variance computation (subtraction),
	#   variance sums close to zero sometimes occur as small negative numbers
	#  => Zero clamping
	if curr_node.sq_err_sum < 0.0:
		curr_node.sq_err_sum = 0.0

	for c in curr_node.funcs_called:
		if c != None:
			finish_variance_computation_recursive(c)




significance_fraction = 0.0002
max_tolerable_average_deviation = 0.000001
enable_check_if_means_stable = True
def check_if_means_stable(func_node, current_context, current_count):
	"""Check if means have stabilized in cet subtree under func_node"""

	current_min_significant_exec_time = current_count * significance_fraction

	def recursive_get_significant_mean_deviations(func_node):

		node_context = func_node.get_call_chain_context()
		#print "recursive_get_significant_mean_deviations(): node_context = ", node_context

		len_node_context = len(node_context)
		len_current_context = len(current_context)
		if len_node_context <= len_current_context:
			node_is_in_execution_context = True
			for i in range(len_node_context):
				if node_context[len_node_context-1 - i] != current_context[len_current_context-1 - i]:
					node_is_in_execution_context = False
					break
		else:
			node_is_in_execution_context = False

		# node_is_in_execution_context == False <=> current node has completed all prior invocations
		# node_is_in_execution_context == True <=> current node has been entered but not exited yet

		if (not node_is_in_execution_context) and func_node.total_count < current_min_significant_exec_time:
			return (0.0, 0) # insignificant subtree would not contribute to deviation

		#Currently executing nodes (i.e., those with node_is_in_execution_context == True) may not yet have
		# been updated with their significant execution times. Therefore, to be safe, we want all currently
		# executing nodes (whether significant or not) to be recursively examined for significant children.

		if node_is_in_execution_context:
			completion_count = func_node.invoke_count - 1
		else:
			completion_count = func_node.invoke_count

		if completion_count == 0 or func_node.total_count < current_min_significant_exec_time:
			# no profile data available yet, or it is insignificant
			deviation = 0.0
			num_nodes_in_deviation = 0

		else: # completion_count > 0 => some profile data available
			curr_mean = float(func_node.total_count) / float(completion_count)
			if func_node.analysis.has_key("last_mean"):
				last_mean = func_node.analysis["last_mean"]
				deviation = abs(last_mean - curr_mean) / curr_mean

			else: # cannot calculate deviation as yet, assume the worst
				deviation = 1.0

			num_nodes_in_deviation = 1

			func_node.analysis["last_mean"] = curr_mean

		for c in func_node.funcs_called:
			if c != None:
				(c_dev, c_num_dev) = recursive_get_significant_mean_deviations(c)
				deviation += c_dev
				num_nodes_in_deviation += c_num_dev

		#print "(deviation, num_nodes_in_deviation) = ", (deviation, num_nodes_in_deviation)
		return (deviation, num_nodes_in_deviation)

	(deviation, num_nodes_in_deviation) = recursive_get_significant_mean_deviations(func_node)

	print "check_if_means_stable(): (deviation, num_nodes_in_deviation) = ", (deviation, num_nodes_in_deviation), \
			" for current_count = ", current_count

	if num_nodes_in_deviation > 0:
		average_deviation = deviation / num_nodes_in_deviation
	else:
		average_deviation = 0.0

	if average_deviation < max_tolerable_average_deviation:
		return True # appears to be stabilized

	return False # not stabilized




profile_re = re.compile('(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+(\d+)')

import time

#num_profile_cutoff_events = 0
profile_stats_check_steps = 7500000
force_min_profile_steps =   30000000
force_max_profile_steps =   70000000
#pass_number = 0
def do_profile_pass(profile_file_name):
	global pass_number
	pass_number += 1

	common.MARK_PHASE_START("PROFILE PASS # " + str(pass_number))
	profile_handle = open(profile_file_name, "r")

	# simulate entry into main (mimic process_entry_event())
	global profile_cet
	if pass_number == 1:
		profile_cet = FuncNode(map_func_name_to_id["main"], "main")
		profile_cet.invoke_count = 1

	elif pass_number == 2:
		profile_cet.children_exec_counts = len(profile_cet.funcs_called) * [0]

	elif pass_number >= 3:
		profile_cet.children_exec_counts = len(profile_cet.funcs_called) * [0]

	profile_cet.last_entry_count = 0

	global num_profile_cutoff_events

	global curr_node
	curr_node = profile_cet

	global lineno
	lineno = 0
	max_count = 0
	file_counter = 1
	continue_reading = True
	while continue_reading:
		for line in profile_handle:
			lineno += 1

			try:
				func_id, func_lexical_id, loop_lexical_id, prof_event_type, count \
						= profile_re.match(line).group(1,2,3,4,5)
			except:
				sys.exit("do_profile_pass(): syntax error in " + profile_file_name + " at lineno = " + str(lineno) \
					+ " line = " + str(line))

			func_id = int(func_id)
			func_lexical_id = int(func_lexical_id)
			loop_lexical_id = int(loop_lexical_id)
			count = int(count)

			#print "EVENT: func_id =", func_id, "func_lexical_id =", func_lexical_id, \
			#		"loop_lexical_id =", loop_lexical_id, prof_event_type, "count =", count

			process_profile_event(func_id, func_lexical_id, loop_lexical_id, prof_event_type, count)
			max_count = count

			# Terminate if statistics stabilized, or force_max_profile_steps reached
			if pass_number == 1:
				if force_max_profile_steps != None and lineno >= force_max_profile_steps:
					num_profile_cutoff_events = lineno
					print "PROFILE TERMINATED: force_max_profile_steps reached. num_profile_cutoff_events = ", num_profile_cutoff_events
					continue_reading = False
					break

				if count > 0 and lineno >= force_min_profile_steps and lineno % profile_stats_check_steps == 0:
					current_context = curr_node.get_call_chain_context()
					print "Time elapsed = ", time.time() - common.start_time
					print "num_profile_events = ", lineno, " current_context = ", current_context

					if enable_check_if_means_stable == True and check_if_means_stable(profile_cet, current_context, count):
						num_profile_cutoff_events = lineno
						print "MEANS Stabilized: num_profile_cutoff_events = ", num_profile_cutoff_events
						continue_reading = False
						break

			elif lineno == num_profile_cutoff_events: # all subsequent passes
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
	while curr_node.func_name != "main":
		print "Simulating exit: func_id = %s func_lexical_id = %s func_name = %s" \
				% (curr_node.func_id, curr_node.func_lexical_id, curr_node.func_name)

		process_profile_event(curr_node.func_id, curr_node.func_lexical_id, curr_node.loop_lexical_id, "exit", max_count)

	if(curr_node.func_name != "main"):
		sys.exit("ERROR: Read profile is incomplete. curr_node is not 'main'")

	process_profile_event(map_func_name_to_id["main"], None, None, "exit", max_count)

	common.MARK_PHASE_COMPLETION("PROFILE PASS # " + str(pass_number))
	return max_count



def read_profile(profile_file_name = "profile.dump", construct_covariance = True):
	'''Constructs a fresh profile representation from given profile events dump file,
	assuming that 'func_id_to_name', 'map_func_name_to_id', 'func_id_to_info' and 'map_func_name_to_loop_hier'
	are globals in the profile module that are already defined'''

	global profile_cet
	profile_cet = None

	global pass_number
	pass_number = 0

	global num_profile_cutoff_events
	num_profile_cutoff_events = 0

	global map_called_NULL_FUNC_artificial_lexical_id
	map_called_NULL_FUNC_artificial_lexical_id = {}

	max_count = do_profile_pass(profile_file_name) # Pass 1: construct CET, determine 'total_count' and 'invoke_count'
	finish_variance_computation_recursive(profile_cet)

	if construct_covariance == True:
		do_profile_pass(profile_file_name) # Pass 2
	else:
		pass_number += 1

	return profile_cet, max_count, num_profile_cutoff_events, map_called_NULL_FUNC_artificial_lexical_id, pass_number

#profile_cet, max_count, num_profile_cutoff_events = read_profile()


def construct_significant_node_lists(func_node):
	if func_node.total_count >= min_significant_exec_time:
		func_id = func_node.func_id
		func_id_to_info[func_id].significant_nodes.append(func_node)

		for c in func_node.funcs_called:
			if c != None:
				construct_significant_node_lists(c)




##################################################
################ Profile User API ################
##################################################

class ProfileRepresentation:
	"Container: Represents a single application's profile information read from associated profile dump files"

	def __init__(self,
			profile_file_name          = None,
			func_id_to_name            = None, 
			map_func_name_to_id        = None,
			func_id_to_info            = None,
			map_func_name_to_loop_hier = None,
			profile_cet                = None,
			max_count                  = None,
			num_profile_cutoff_events  = None,
			min_significant_exec_time  = None,
			map_called_NULL_FUNC_artificial_lexical_id = None,
			pass_number                = None
		):

			self.profile_file_name         = profile_file_name
				# file name containing profile events

			self.func_id_to_name            = func_id_to_name
				# array: func_name (string) indexed by func_id (positive integer)

			self.map_func_name_to_id        = map_func_name_to_id
				# dict: reverse lookup of func_id using func_name

			self.func_id_to_info            = func_id_to_info
				# array of instances of FuncInfo, indexed by corresponding func_ids

			self.map_func_name_to_loop_hier = map_func_name_to_loop_hier
				# map: provides the lexical loop-nesting scopes in body of a given function
				# value format = (0, [(1, [...]), (2, [...])) where '0' is function scope containing
				#    adjacent loops '1' and '2' which have their own nesting structure

			self.profile_cet                = profile_cet
				# represents the dynamic profile of the application in CET format
				#    (profile_cet = instance of FuncNode for 'main', which then recursively contains application)

			self.max_count                  = max_count
				# maximum cycle count achieved before processing of the profile was terminated

			self.num_profile_cutoff_events  = num_profile_cutoff_events
				# number of events processed after which reading of the profile was terminated

			self.min_significant_exec_time = min_significant_exec_time
				# minimum number of cycles required for a FuncNode to be considered significant

			self.map_called_NULL_FUNC_artificial_lexical_id = map_called_NULL_FUNC_artificial_lexical_id
				# For each function-name called via a NULL_FUNC, assigns a unique ID that will
				#  serve as the func_lexical_id of any instance of that function called from under any NULL_FUNC

			self.pass_number                = pass_number
				# gives number of passes made over profile events dump file so far 


	def read(self, func_map_file_name, func_loop_hier_file_name, profile_file_name, construct_covariance = True, specified_force_max_profile_steps = False, specified_enable_check_if_means_stable = True):
		"Construct profile representation by reading profile dump files"

		self.profile_file_name = profile_file_name

		(self.func_id_to_name, self.map_func_name_to_id, self.func_id_to_info) = read_func_map(func_map_file_name)

		self.map_func_name_to_loop_hier = read_func_loop_hier(func_loop_hier_file_name,
				need_special_NULL_FUNC = (self.map_func_name_to_id.has_key("NULL_FUNC") == True))


		global func_id_to_name, map_func_name_to_id, func_id_to_info, map_func_name_to_loop_hier

		func_id_to_name, map_func_name_to_id, func_id_to_info, map_func_name_to_loop_hier = \
			self.func_id_to_name, self.map_func_name_to_id, \
			self.func_id_to_info, self.map_func_name_to_loop_hier

		global force_max_profile_steps
		if specified_force_max_profile_steps != False:
				#user specified a value (None ==> unbounded, o.w. the desired count)
			assert specified_force_max_profile_steps == None or type(specified_force_max_profile_steps) == type(1) #int type
			force_max_profile_steps = specified_force_max_profile_steps

		global enable_check_if_means_stable
		assert specified_enable_check_if_means_stable == True or specified_enable_check_if_means_stable == False
		enable_check_if_means_stable = specified_enable_check_if_means_stable

		(self.profile_cet, self.max_count, self.num_profile_cutoff_events, \
				self.map_called_NULL_FUNC_artificial_lexical_id, self.pass_number) \
					= read_profile(profile_file_name, construct_covariance)

		self.min_significant_exec_time = self.max_count * significance_fraction

		global min_significant_exec_time
		min_significant_exec_time = self.min_significant_exec_time
		construct_significant_node_lists(self.profile_cet)

		global map_called_NULL_FUNC_artificial_lexical_id
		map_called_NULL_FUNC_artificial_lexical_id = self.map_called_NULL_FUNC_artificial_lexical_id

		return self

#profile_repr = ProfileRepresentation()
#profile_repr.read("func_map.dump", "func_loop_hier.dump", "profile.dump")

#max_count = profile_repr.max_count


#print "PROFILE_CET = ", profile_cet.__repr__()



def run_user_profile_pass(profile_repr, po_visit_func):
	'''Make another traversal pass on the CET representation driven by the associated profile event data.
		Parms:
			profile_repr: an instance of ProfileRepresentation, contains enough info to enable the traversal

			po_visit_func: a function that takes (curr_node, count_diff, children_exec_counts) as inputs.
				po_visit_func() is invoked when a FuncNode given by 'curr_node' is exited (i.e. post-order visitation)
				during profile-driven traversal of the CET.

				'count_diff' gives the number of cycles consumed hierarchically by current invocation of 'curr_node'
				in the profile.

				'children_exec_counts' is a list (with elements corresponding to elements of 'curr_node.funcs_called')
				that gives the total cycles consumed by each child during the current invocation of 'curr_node'.

			The user should appropriately define po_visit_func() to perform pass-specific analysis on the CET
	'''

	global func_id_to_name, map_func_name_to_id, func_id_to_info, map_func_name_to_loop_hier, profile_cet, \
		num_profile_cutoff_events, map_called_NULL_FUNC_artificial_lexical_id, pass_number

	func_id_to_name            = profile_repr.func_id_to_name
	map_func_name_to_id        = profile_repr.map_func_name_to_id
	func_id_to_info            = profile_repr.func_id_to_info
	map_func_name_to_loop_hier = profile_repr.map_func_name_to_loop_hier
	profile_cet                = profile_repr.profile_cet
	num_profile_cutoff_events  = profile_repr.num_profile_cutoff_events
	map_called_NULL_FUNC_artificial_lexical_id = profile_repr.map_called_NULL_FUNC_artificial_lexical_id
	pass_number                = profile_repr.pass_number

	global user_post_order_visit_func
	user_post_order_visit_func = po_visit_func

	do_profile_pass(profile_repr.profile_file_name)

	profile_repr.pass_number = pass_number




