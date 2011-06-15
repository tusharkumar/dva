# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com

################ Significant Functions Analysis API ################

def cmp_large_int_exec_time(x, y):
	if y.exec_time > x.exec_time: return 1
	if y.exec_time < x.exec_time: return -1
	return 0


def get_sorted_significant_func_info_list(func_id_to_info, min_significant_exec_time):
	significant_func_list = func_id_to_info[:] # assumes: each id in 0..max_id has a func_name associated with it

	#significant_func_list.sort(lambda x, y: y.exec_time - x.exec_time) #fails for large exec_times
	significant_func_list.sort(cmp_large_int_exec_time)
	sum_exec_time = 0.0
	for i, f in enumerate(significant_func_list):
	#	if f.exec_time > sum_exec_time * 0.10: # each additional function should contribute atleast 10% to sum

		if f.exec_time >= min_significant_exec_time:
			sum_exec_time += f.exec_time
		else:
			significant_func_list[i:] = []
			break

	return significant_func_list

def convert_func_info_to_whatever_map_to_sorted_list(func_info_to_whatever_map):
	func_info_tuples_list = []
	for f in func_info_to_whatever_map:
		func_info_tuples_list.append( (f, func_info_to_whatever_map[f]) )

	func_info_tuples_list.sort( lambda tuple1, tuple2: cmp_large_int_exec_time(tuple1[0], tuple2[0]) )

	return func_info_tuples_list




################ Boundedness Analysis API ################

def determine_bounded_recursive(func_node, map_func_name_to_loop_hier, min_significant_exec_time):
	if func_node.total_count < min_significant_exec_time:
		return

	for c in func_node.funcs_called:
		if c != None:
			determine_bounded_recursive(c, map_func_name_to_loop_hier, min_significant_exec_time)

	print func_node.func_name
	print map_func_name_to_loop_hier[ func_node.func_name ]
	if len(map_func_name_to_loop_hier[ func_node.func_name ][1]) > 0: # contains loops
		func_node.analysis["bounded"] = False
	else: # no direct loops, check if called significant functions are also bounded
		func_node.analysis["bounded"] = True
		for c in func_node.funcs_called:
			if c != None and c.total_count >= min_significant_exec_time:
				if not c.analysis["bounded"]:
					func_node.analysis["bounded"] = False



################ Minimal Distinguishing Call-Context Extraction API ################

def extract_min_distinguishing_context_for_node_sets(match_nodes, other_nodes):
	"""Determine call-chain-contexts for nodes in match_nodes to distinguish against nodes in other_nodes.
	     match_nodes, other_nodes: lists of func_nodes of the same function
	   Return: list [elem, elem, ...], with each elem = (a-match-node, distinguishing-call-chain-context-for-node)
		   with one elem per node in match_nodes
	"""
	distinguishing_context = [] # a list of (node, min-distinguishing-call-chain) tuples

	for m in match_nodes:
		m_call_chain_context = m.get_call_chain_context()

		undistinguished_other_nodes_set = set(other_nodes)

		i = 0 # distinguishing index
		while i < len(m_call_chain_context):
			if len(undistinguished_other_nodes_set) == 0:
				break

			delete_set = set([])
			for uo_node in undistinguished_other_nodes_set:
				if m_call_chain_context[i] != uo_node.get_call_chain_context()[i]:
					delete_set.add(uo_node)

			undistinguished_other_nodes_set -= delete_set
			i += 1

		if len(undistinguished_other_nodes_set) != 0:
			sys.exit("extract_min_distinguishing_context_for_node_sets(): ERROR: distinguishing context not found for FuncNode", m)

		distinguishing_context.append( (m, m_call_chain_context[0:i+1]) )
	
	return distinguishing_context


def extract_min_distinguishing_context_for_function_given_match_criteria(func_info, match_criteria_function):
	"""Determine call-chain-contexts for all significant func_nodes of 'func_info', that satisfy 'match_criteria_function'.
	     Call-chain-contexts will distinguish the matching func_nodes from all other significant func_nodes that
	     do not match 'match_criteria_function'
	"""
	match_nodes = []
	other_nodes = []

	for f in func_info.significant_nodes:
		if match_criteria_function(f):
			match_nodes.append(f)
		else:
			other_nodes.append(f)

	distinguishing_context = extract_min_distinguishing_context_for_node_sets(match_nodes, other_nodes)

	return distinguishing_context





################ Pattern Similarity Trees API ################

def detach_call_chain_context(call_chain):
	new_cc = call_chain[:]
	new_cc[-1] = (new_cc[-1][0], None)
	return new_cc


class PatternLookupStageNode:
	"Multistage lookup tree. In i'th stage lookup i'th link of call-chain"
	def __init__(self, func_name, lexical_id):
		self.func_name = func_name
		self.lexical_id = lexical_id

		self.next_stage_list = []
		self.curr_stage_matching_func_nodes_list = []

	def lookup(self, call_chain):
		"Return list of matching func_nodes"

		if (self.func_name, self.lexical_id) != call_chain[0]:
			return [] # no matches

		#this stage matched
		if len(call_chain) == 1: #This is the end of the call-chain
			return self.curr_stage_matching_func_nodes_list

		#call-chain is longer
		for nsc in self.next_stage_list:
			ret_list = nsc.lookup(call_chain[1:])
			if len(ret_list) != 0:
				return ret_list

		return []

	def add(self, m, call_chain):
		# Assumption: call_chain[0] already matches (self.func_name, self.lexical_id)

		if (self.func_name, self.lexical_id) != call_chain[0]:
			sys.exit("PatternLookupStageNode.add(): ERROR: node (func_name = %s, lexical_id = %s) does not match call-chain = %s"
					% (self.func_name, self.lexical_id, call_chain))

		if len(call_chain) == 1: #call-chain ends here
			self.curr_stage_matching_func_nodes_list.append(m)
			return

		# len(call_chain) >= 2
		match_loc = None
		for i, nsc in enumerate(self.next_stage_list):
			if (nsc.func_name, nsc.lexical_id) == call_chain[1]:
				match_loc = i
				break

		if match_loc == None:
			self.next_stage_list.append( PatternLookupStageNode(call_chain[1][0], call_chain[1][1]) )
			match_loc = len(self.next_stage_list) - 1

		self.next_stage_list[match_loc].add(m, call_chain[1:])
		


class MultiStagePatternLookup:
	"Wrapper around tree of PatternLookupStageNode"
	def __init__(self):
		self.start_stage_node = PatternLookupStageNode(None, None) #dummy node, that will be root of tree

	def lookup(self, call_chain):
		return self.start_stage_node.lookup([(None, None)] + call_chain)

	def add(self, m, call_chain):
		self.start_stage_node.add(m, [(None, None)] + call_chain)


def make_pattern_groups_of_identical_call_chain_contexts(pattern_list):
	'''Take input ('pattern_list') a list of elements of form: (m, call-chain-context-for-m),
		Produce a list whose each element is a group of patterns with identical call-chains-contexts
		taken from 'pattern_list'
	'''
	# Want to map a call-chain to all to nodes 'm' where it occurs in pattern_list
	# Like a map of form cc -> [m1, m2, ...]

	call_chain_lookup_tree = MultiStagePatternLookup()
	list_of_unique_call_chains = []

	for m, cc in pattern_list:
		#print "@@@ Examining (%s, %s) for lookup" % (m, cc)
		if len(call_chain_lookup_tree.lookup(cc)) == 0: # call-chain doesn't exist so far
			list_of_unique_call_chains.append(cc)

		call_chain_lookup_tree.add(m, cc)

	#print "@@@ list_of_unique_call_chains = ", list_of_unique_call_chains

	identical_groups = []
	for cc in list_of_unique_call_chains:
		identical_groups.append( [(m, cc) for m in call_chain_lookup_tree.lookup(cc)] )

	return identical_groups



class PatternSimilarityTreeNode:
	def __init__(self, group = [], parent = None, children = []):
		'''group is an collection of identical call-chain patterns,
			parent and children are corresponding PatternSimilarityTreeNodes'''
		self.group = group
		self.parent = parent
		self.children = children

	def check_similarity(self, sm_func):
		"Return true if all patterns in group are identical based on sm_func"
		for i, p1 in enumerate(self.group): #compare i'th element with all elements after
			for j in range(i+1, len(self.group)):
				p2 = self.group[j]
				if not sm_func(p1, p2):
					return False

		return True #also applies if self.group has 0 or 1 entries

	def grow_and_differentiate(self):
		'''Grow by 1 the call-chain in each pattern,
			and create new children nodes if patterns become distinguishable'''
		if len(self.group) <= 1:
			return #nothing to differentiate

		curr_call_chain_len = len(self.group[0][1]) #assumes that group has atleast one element
		new_len = curr_call_chain_len + 1
			#assumption: if there are multiple identical patterns, then we can certainly grow them by 1
			# (else they would already be distinguishable, since all calling-contexts are distinguishable at main)
		grown_group = [(m, detach_call_chain_context(m.get_call_chain_context()[0:new_len])) for (m, old_ccc) in self.group]

		child_groups = make_pattern_groups_of_identical_call_chain_contexts(grown_group)

		self.children = []
		for c in child_groups:
			self.children.append( PatternSimilarityTreeNode(c, self) )

	def construct_tree(self, sm_func):
		if not self.check_similarity(sm_func):
			self.grow_and_differentiate() #create children PatternSimilarityTreeNodes
			for c in self.children:
				c.construct_tree(sm_func)

	def get_leaf_groups(self):
		"Assumes that construct_tree() has already been applied"
		if len(self.children) == 0:
			return [self.group]

		all_children_leaf_groups = []
		for c in self.children:
			all_children_leaf_groups += c.get_leaf_groups()

		return all_children_leaf_groups


def detach_and_differentiate_using_pst_groups(distinguishing_context_map, sm_func):
	"Update distinguishing_context_map by suitably lengthning patterns using Similarity-Measure 'sm_func'"
	for f in distinguishing_context_map.copy():
		if len(distinguishing_context_map[f]) == 0:
			continue

		#print "##--## Generating PSTs for Function ", f

		context_list_with_detached_contexts = [(m, detach_call_chain_context(cc)) for m, cc in distinguishing_context_map[f]]

		pattern_groups_with_identical_ccc = make_pattern_groups_of_identical_call_chain_contexts(context_list_with_detached_contexts)
		#print " ^^ Pattern groups found:"
		#print pattern_groups_with_identical_ccc

		distinguishing_context_map[f] = []
		for identical_ccc_pattern_group in pattern_groups_with_identical_ccc:
			pst = PatternSimilarityTreeNode(identical_ccc_pattern_group)
			pst.construct_tree(sm_func)
			pst_leaf_groups = pst.get_leaf_groups()

			#print " ##### group %s divided into PST leaf-groups %s" % (identical_ccc_pattern_group, pst_leaf_groups)

			for lg in pst_leaf_groups:
				distinguishing_context_map[f].extend(lg)



