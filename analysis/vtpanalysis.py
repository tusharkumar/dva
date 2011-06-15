#!/usr/bin/python

# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com


import math
import common
import sys

# Command-Line Arguments extraction
def extract_numerical_argument(arg, assignment_pattern):
	'''Returns a numerical value if 'arg' contains assignment_pattern with an actual numerical value provided.
	Returns "invalid" if 'arg' contains assignment_pattern, but does not have a valid numerical value specified.
	Returns None if 'arg' does not contain assignment_pattern.
	'''
	numerical_value_is_invalid = False
	extracted_numerical_value = None
	if arg[0:len(assignment_pattern)] == assignment_pattern: #matching argument found
		try:
			extracted_numerical_value = float( arg[len(assignment_pattern):] )
		except ValueError:
			numerical_value_is_invalid = True
	
	if numerical_value_is_invalid == False:
		return extracted_numerical_value
	else:
		return "invalid"
	
setting_merge_cost_limit = None
setting_vim_cutoff_fraction = None
setting_force_max_profile_steps = None
setting_enable_check_if_means_stable = None
extracted_trees_filename = None

valid_arguments = True
if len(sys.argv) > 5:
	valid_arguments = False
unconsumed_arguments = sys.argv[1:] #Could be [] if none specified

if valid_arguments and len(unconsumed_arguments) >= 1:
	last_arg = unconsumed_arguments[len(unconsumed_arguments) - 1]

	setting_enable_check_if_means_stable = extract_numerical_argument(last_arg, "enable_check_if_means_stable=")
	if setting_enable_check_if_means_stable == "invalid" \
			or (setting_enable_check_if_means_stable != None and setting_enable_check_if_means_stable != -1 and setting_enable_check_if_means_stable != 1):
		print "Argument Error: enable_check_if_means_stable must be either -1 or 1"
		valid_arguments = False

	if setting_enable_check_if_means_stable != None:
		unconsumed_arguments.pop() #consume last argument

if valid_arguments and len(unconsumed_arguments) >= 1:
	last_arg = unconsumed_arguments[len(unconsumed_arguments) - 1]

	setting_force_max_profile_steps = extract_numerical_argument(last_arg, "force_max_profile_steps=")
	if setting_force_max_profile_steps == "invalid":
		print "Argument Error: force_max_profile_steps must be specified a numerical value"
		valid_arguments = False

	if setting_force_max_profile_steps != None:
		unconsumed_arguments.pop() #consume last argument

if valid_arguments and len(unconsumed_arguments) >= 1:
	last_arg = unconsumed_arguments[len(unconsumed_arguments) - 1]

	setting_vim_cutoff_fraction = extract_numerical_argument(last_arg, "vim_cutoff_fraction=")
	if setting_vim_cutoff_fraction == "invalid" \
			or (setting_vim_cutoff_fraction != None and (setting_vim_cutoff_fraction < 0.0 or setting_vim_cutoff_fraction > 1.0)):
		print "Argument Error: vim_cutoff_fraction must be specified a numerical value between 0.0 and 1.0\n"
		valid_arguments = False

	if setting_vim_cutoff_fraction != None:
		unconsumed_arguments.pop() #consume last argument

if valid_arguments and len(unconsumed_arguments) >= 1:
	last_arg = unconsumed_arguments[len(unconsumed_arguments) - 1]

	setting_merge_cost_limit = extract_numerical_argument(last_arg, "merge_cost_limit=")
	if setting_merge_cost_limit == "invalid":
		print "Argument Error: merge_cost_limit must be specified a numerical value"
		valid_arguments = False

	if setting_merge_cost_limit != None:
		unconsumed_arguments.pop() #consume last argument

if valid_arguments and len(unconsumed_arguments) >= 1:
	if len(unconsumed_arguments) > 1:
		valid_arguments = False
	else:
		extracted_trees_filename = unconsumed_arguments.pop()

if valid_arguments == False:
	sys.exit( \
'''Usage:
    vtpanalysis.py [<extracted_trees_filename>] [cost_limit=<value between -inf to +inf, though usually between 0.0 and 1.0] \
			[vim_cutoff_fraction=<value between 0.0 and 1.0>] [force_max_profile_steps=<-1 for unbounded or a positive integral desired bound>] [enable_check_if_means_stable=<-1 to disable, 1 to enable>]

    <extracted_trees_filename> contains a dump of a list of ExtractTree data structures (defined in rttaskanalysis.py).
    If optional argument <extracted_trees_filename> is not specified,
      rttaskanalysis.py is invoked on 'profile.dump' to generate an 'extract_trees.txt' file
      which is then used as the default value for <extracted_trees_filename>

    If other optional arguments are not specified, they assume default value.
'''
	)

print "--- Arguments"
print "   extracted_trees_filename =", extracted_trees_filename
print "   setting_merge_cost_limit =", setting_merge_cost_limit
print "   setting_vim_cutoff_fraction =", setting_vim_cutoff_fraction
print "   setting_force_max_profile_steps =", setting_force_max_profile_steps
print "   setting_enable_check_if_means_stable =", setting_enable_check_if_means_stable

import rttaskanalysis
from rttaskanalysis import ExtractTree

if extracted_trees_filename != None: #<extracted_trees_filename> argument specified
	print "--- Reading extracted trees from %s ---" % (extracted_trees_filename, )

	etfhandle = open(extracted_trees_filename)

	extracted_trees_string = ""
	for line in etfhandle:
		extracted_trees_string += line

	exec("extracted_trees = " + extracted_trees_string) # extracted_trees defined now
	print "extracted_trees = ", extracted_trees


else:
	force_max_profile_steps = False #use defaults
	if setting_force_max_profile_steps != None:
		if setting_force_max_profile_steps == -1:
			force_max_profile_steps = None #force unbounded
		elif type(setting_force_max_profile_steps) == type(1.0) and setting_force_max_profile_steps > 0:
			force_max_profile_steps = int(setting_force_max_profile_steps)
		else:
			sys.exit("vtpanalysis: ERROR: force_max_profile_steps argument must be either -1 or a positive integer")
	enable_check_if_means_stable = True
	if setting_enable_check_if_means_stable == -1:
		enable_check_if_means_stable = False
	print "--- Running rttaskanalysis.run_analysis() to produce extracted trees ---"
	rttaskanalysis.run_analysis(force_max_profile_steps, enable_check_if_means_stable)
	extracted_trees = rttaskanalysis.extracted_trees
	print "--- Completed rttaskanalysis.run_analysis() ---"


#FIXME: To dos
# 1) Implement merge-cost-function
# 2) Allow Tree-Structure merging (pass 1) for subtrees where the call-context-sets do not match up exactly. Maybe allow one underlying node to have a superset cccs of the other nodes cccs, or even just different cccss'.
#    This is important where internal merging before merging across separate patterns *prevents* merging because the cccs no longer match exactly.
#     This leads to more pattern-trees remaining than the case where no internal merging was attempted.
#   Examples:
#     - mediabench2_video/mpeg2enc/mpeg2/regr_decode: mergers between [5] and [113] instances of "Headers.B1"
#     - mediabench2_video/mpeg2enc/mpeg2/regr_encode: mergers between [2] and [43] instances of "frame_estimate.10"
#    Instead, roll differences between merged cccss' into *merge-cost*.
#   FIXME locations:
#     - Tree structure algo, comparing cccs
#     - Internal merging algo, performing unions of cccs of merged siblings



#####################################
### VTPatternNode and dumping
#####################################

import distribution
#FIXME:
# 1) Include distribution into class Statistics:
#    - Replace: invoke_count, mean, cov -> list of (mean, std, count)
#    - Additional field for "task": contribution_distribution = a list of (contrib_fraction_mean, contrib_fraction_std, contrib_fraction_count)
#       s.t. elements of task.contribution_distribution correspond to elements of task.children.
#
#       where: initially contrib_fraction_mean = contrib_fraction from contributor, contrib_fraction_std = 0.0, contrib_fraction_count = invoke_count of contributor
#        and children that are not contributors or are None are treated as contributing zero.
#
# 2) Before contribution_distribution can be compared between task1 and task2 merged into m_task, we will need to "normalize" task1.contribution_distribution
#      and task2.contribution_distribution so that corresponding terms match up to elements of m_task.children

#class Statistics:
#	def __init__(self, total_count = None, invoke_count = None, mean = None, cov = None, contributesVariance = None, contrib_fraction = None):
#		self.total_count         = total_count
#		self.invoke_count        = invoke_count
#		self.mean                = mean
#		self.cov                 = cov
#		if mean != None and cov != None:
#			self.std = cov * mean
#		else:
#			self.std = None
#		self.contributesVariance = contributesVariance
#		self.contrib_fraction    = contrib_fraction
#
#	def clone(self):
#		return Statistics(self.total_count, self.invoke_count, self.mean, self.cov, self.contributesVariance, self.contrib_fraction)

class ContribStatistics:
	def __init__(self, exec_time_distr_wrt_variant_task,
			weighted_sum_of_underlying_contrib_fractions, weighted_squared_sum_of_underlying_contrib_fractions):
		self.exec_time_distr_wrt_variant_task = exec_time_distr_wrt_variant_task.clone()
			# gaussian mixture density for the cumulative execution time of a contributor per invocation of the corresponding variant task.
			# An instance of Distribution

		self.weighted_sum_of_underlying_contrib_fractions = weighted_sum_of_underlying_contrib_fractions
		self.weighted_squared_sum_of_underlying_contrib_fractions = weighted_squared_sum_of_underlying_contrib_fractions

	def clone(self):
		return ContribStatistics( self.exec_time_distr_wrt_variant_task,
				self.weighted_sum_of_underlying_contrib_fractions, self.weighted_squared_sum_of_underlying_contrib_fractions )

	def __repr__(self):
		if NodeStatistics.enable_reduced_distribution_printing == True:
			exec_time_distr_wrt_variant_task_print = self.exec_time_distr_wrt_variant_task.get_combined_mean_std_count()
		else:
			exec_time_distr_wrt_variant_task_print = self.exec_time_distr_wrt_variant_task
		return "(%s, %s, %s)" % (exec_time_distr_wrt_variant_task_print, self.weighted_sum_of_underlying_contrib_fractions, self.weighted_squared_sum_of_underlying_contrib_fractions)


class NodeStatistics:
	'''Basic statistics apply to nodes of all types: task/contrast/contributor.
	However, contribution_stats should be defined only for a task node'''
	enable_reduced_distribution_printing = False
	enable_vim_printing = False # Variance Impact Metric

	def __init__(self, total_count = None, exec_time_distr = None, contribution_stats = None):
		self.total_count = total_count

		self.exec_time_distr = exec_time_distr
			#gaussian mixture density for node execution time: an instance of Distribution

		self.contribution_stats = []
			#list of instances of ContribStatistics providing information for contributors to this node.
			# list elements correspond to elements of node's originating_linear_segments.
		if contribution_stats != None:
			self.contribution_stats = [cs.clone() for cs in contribution_stats]

	def add_contributor_info(self, contrib_index, exec_time_distr_wrt_variant_task, \
			weighted_sum_of_underlying_contrib_fractions, weighted_squared_sum_of_underlying_contrib_fractions):
		'''Should be invoked only for a task node'''
		if len(self.contribution_stats) <= contrib_index:
			self.contribution_stats.extend( [None] * (contrib_index+1 - len(self.contribution_stats)) )
		assert self.contribution_stats[contrib_index] == None
		self.contribution_stats[contrib_index] = ContribStatistics(exec_time_distr_wrt_variant_task,
				weighted_sum_of_underlying_contrib_fractions, weighted_squared_sum_of_underlying_contrib_fractions)

	def get_contrib_fraction(self, i):
		'''Should be invoked only for a task node'''
		(contrib_cumm_mean, contrib_cumm_std, contrib_cumm_count) = self.contribution_stats[i].exec_time_distr_wrt_variant_task.get_combined_mean_std_count()
		(overall_mean, overall_std, node_count) = self.exec_time_distr.get_combined_mean_std_count()
		assert node_count == contrib_cumm_count

		contrib_fraction_i = (contrib_cumm_std * contrib_cumm_std) / (overall_std * overall_std)
		return contrib_fraction_i

	def get_vim(self): #variance impact metric
		(mean, std, count) = self.exec_time_distr.get_combined_mean_std_count()
		vim = std * count
		return vim

	def get_weight(self):
		(mean, std, count) = self.exec_time_distr.get_combined_mean_std_count()
		weight = mean * count
		return weight

	def clone(self):
		return NodeStatistics(self.total_count, self.exec_time_distr, self.contribution_stats)

	def __repr__(self):
		if NodeStatistics.enable_reduced_distribution_printing == True:
			exec_time_distr_print = self.exec_time_distr.get_combined_mean_std_count()
		else:
			exec_time_distr_print = self.exec_time_distr

		if NodeStatistics.enable_vim_printing == True:
			vim = self.get_vim()
			return "(%s, vim=%s cs=%s)" % (exec_time_distr_print, vim, self.contribution_stats)

		return "(%s, cs=%s)" % (exec_time_distr_print, self.contribution_stats)
			



class VTPatternNode:
	enable_stats_printing = True

	def __init__(self, extract_tree_root = None):
		'''Construct with extract_tree_root != None, when initializing from an existing instance of ExtractTree,
			else invoke without parameters to create standalone VTPatternNode'''
		self.corresponding_nodes = [] #list of references to ExtractTree or VTPatternNode instances in lower-level patterns

		self.func_name = None

		self.node_type = None #"task", "contributor" or "contrast" if defined.
		                      # Access only via get_node_type() method

		self.parent = None


		self.children = [] # list of VTPatternNode instances

		self.children_call_context_sets = [] # list with elements corresponding to elements of 'children'.
			# Each element is a list of non-duplicate call-chain contexts from the child VTPatternNode to the current node.


		self.containing_linear_segment = None #linear segment (if any) that this node is part of, but is not the originator for
		self.originating_linear_segments = [] #list of linear-segments originating at this node. Each element is a list giving the
		                                        # component VTPatternNodes, starting from the current VTPatternNode (which is a "task"),
		                                        # through zero or more "contrast" VTPatternNodes to the terminating "contributor".

		self.stats = None

		if extract_tree_root != None:
			self.func_name = extract_tree_root.get_func_name()
			self.node_type = extract_tree_root.get_node_type()
			self.corresponding_nodes.append(extract_tree_root)
			self.stats = NodeStatistics(
					extract_tree_root.total_count,
					distribution.Distribution( [(extract_tree_root.mean, extract_tree_root.cov * extract_tree_root.mean, extract_tree_root.invoke_count)] ) )
			#Note: self.stats.contribution_stats have not been initialized yet, since construct_linear_segments() has not yet been invoked to construct the contribution structure

			for cc, et_child in extract_tree_root.extracted_children_tuples:
				vt_child = VTPatternNode(et_child)
				vt_child.parent = self
				self.children.append(vt_child)
				self.children_call_context_sets.append([cc])

	def construct_linear_segments(self):
		def visit_vt_node(curr_vt_node, last_parent_task_vt_node):
			print "visit_vt_node(): ENTER: " + curr_vt_node.get_func_name() + " " + curr_vt_node.get_node_type() + " " + curr_vt_node.get_HACK_stats().__repr__()
			if last_parent_task_vt_node == None: #Currently not inside a linear-segment, look for starting points
				if curr_vt_node.get_node_type() == "task": #start new linear-segment
					next_parent_task_vt_node = curr_vt_node
				else:
					next_parent_task_vt_node = None

			else: #already started a linear-segment, looking to end it
				node_type_curr_vt_node = curr_vt_node.get_node_type()
				if node_type_curr_vt_node == "contributor" or node_type_curr_vt_node == "task": # terminate search for end
					if node_type_curr_vt_node == "task": # restart next linear-segment
						next_parent_task_vt_node = curr_vt_node
					else:
						next_parent_task_vt_node = None

					if curr_vt_node.get_contributesVariance() == True: #linear-segment exists
						linear_segment = []
						vt_iter = curr_vt_node
						while vt_iter != last_parent_task_vt_node:
							vt_iter.containing_linear_segment = linear_segment
							linear_segment.insert(0, vt_iter) #modify the same list-object
							vt_iter = vt_iter.parent
						linear_segment.insert(0, last_parent_task_vt_node)

						if linear_segment in last_parent_task_vt_node.originating_linear_segments: #detect duplicates
							sys.exit("VTPatternNode::construct_linear_segments::visit_vt_node(): ERROR: duplicate linear-segment:" \
										+ "\nprocessing curr_vt_node = " + curr_vt_node.__repr__() \
										+ "\nlinear-segment = " + common.list_repr_indented(linear_segment) + "\n" \
										+ "\noriginating_linear_segments = " + common.list_repr_indented(last_parent_task_vt_node.originating_linear_segments)
								)
						last_parent_task_vt_node.originating_linear_segments.append(linear_segment)
					
				else: # continue looking
					next_parent_task_vt_node = last_parent_task_vt_node

			for child_vt_node in curr_vt_node.children:
				visit_vt_node(child_vt_node, next_parent_task_vt_node)

			# Now, The contribution structure within subtree rooted at curr_vt_node is fully constructed

			#Initialize contribution_stats
			for i, linseg in enumerate(curr_vt_node.originating_linear_segments):
				assert curr_vt_node.get_node_type() == "task"

				contrib_node = linseg[-1]
				assert contrib_node.get_contributesVariance() == True, "curr_vt_node = %s \n contrib_node = %s" % (curr_vt_node, contrib_node)

				assert len(contrib_node.corresponding_nodes) == 1
				contrib_extract_tree_node = contrib_node.corresponding_nodes[0]

				assert contrib_extract_tree_node.get_contributesVariance() == True
				contrib_fraction = contrib_extract_tree_node.contrib_fraction
				mean_under_task  = contrib_extract_tree_node.mean_under_task

				assert len(curr_vt_node.stats.exec_time_distr.mean_std_count_list) == 1 # since should have been initialized from a single ExtractTree node
				(task_mean, task_std, task_count) = curr_vt_node.stats.exec_time_distr.mean_std_count_list[0]
				task_variance = task_std * task_std
				
				contrib_variance_under_task = contrib_fraction * task_variance
				contrib_std_under_task = math.sqrt(contrib_variance_under_task)

				curr_vt_node.stats.add_contributor_info(i, distribution.Distribution( [(mean_under_task, contrib_std_under_task, task_count)] ),
						1.0 * contrib_fraction, 1.0 * contrib_fraction * contrib_fraction)


			print "visit_vt_node(): EXIT: " + curr_vt_node.get_func_name() + " " + curr_vt_node.get_node_type()

		#setup recursion
		if self.get_node_type() == "task":
			last_parent_task_vt_node = self
		else:
			last_parent_task_vt_node = None
		visit_vt_node(self, last_parent_task_vt_node)
		return self

	def get_func_name(self):
		return self.func_name

	def get_node_type(self):
		return self.node_type

	def get_HACK_stats(self): #FIXME
		lower_level_corresponding_node = self.corresponding_nodes[0] #pick any one
		return lower_level_corresponding_node.get_HACK_stats()

	def get_contributesVariance(self): #FIXME: needs to be a locally computed amalgam
		lower_level_corresponding_node = self.corresponding_nodes[0] #pick any one
		return lower_level_corresponding_node.get_contributesVariance()

	def get_extract_tree_ids(self):
		et_id_list = []
		for corr in self.corresponding_nodes:
			if corr == None:
				et_id_list.append( [None] )
			else:
				et_id_list.append( corr.get_extract_tree_ids() )
		return et_id_list

	### Correspondences between merged pattern tree structure and the underlying pattern trees being merged
	def get_merged_child_node_corresponding_to_underlying_child_node(self, underlying_child_node):
		'''underlying_child_node is an immediate child node of a node in self.corresponding_nodes.
		Return a node c_mvtn from self.children, such that underlying_child_node is in c_mvtn.corresponding_nodes.
		Error if no such node found.
		'''

		for c_mvtn in self.children:
			if underlying_child_node in c_mvtn.corresponding_nodes:
				return c_mvtn

		sys.exit("VTPatternNode::get_merged_child_node_corresponding_to_underlying_child_node(): " \
				" ERROR: no child of self has underlying_child_node as a corresponding node")
		return None

	def get_merged_ancestor_node_corresponding_to_underlying_ancestor_node(self, underlying_ancestor_node):
		'''underlying_ancestor_node is some ancestor of an underlying node of self, i.e., an ancestor
		of a node in self.corresponding_nodes.
		Return the merged node (some ancestor of self) that has underlying_ancestor_node as its underlying node,
		i.e., return the ancestor of self that contains underlying_ancestor_node in its corresponding_nodes.
		Return None if no such ancestor exists.
		'''
		ancestor_iter = self
		while 1:
			if underlying_ancestor_node in ancestor_iter.corresponding_nodes:
				return ancestor_iter
			ancestor_iter = ancestor_iter.parent
			if ancestor_iter == None:
				return None
		#Assumes (quite reasonably) that parent chain eventually ends, and does not form a cycle in particular
		sys.exit("VTPatternNode::get_merged_ancestor_node_corresponding_to_underlying_ancestor_node(): the impossible occured!")


	def construct_merged_node_sequence_for_underlying_node_sequence(self, underlying_node_sequence):
		'''underlying_node_sequence must be a contiguous sequence in one of the underlying pattern trees of this node,
		and this sequence must start descending from one of self.corresponding_nodes
		'''
		merged_node_sequence = []

		merged_node_iter = self
		underlying_node_iter, i = underlying_node_sequence[0], 0 #assume sequence length of atleast 1
		while 1:
			assert underlying_node_iter in merged_node_iter.corresponding_nodes
			merged_node_sequence.append( merged_node_iter )

			i += 1
			if i == len(underlying_node_sequence):
				break

			next_underlying_node = underlying_node_sequence[i]
			assert next_underlying_node in underlying_node_iter.children #must be a contiguous sequence

			merged_node_iter = merged_node_iter.get_merged_child_node_corresponding_to_underlying_child_node(next_underlying_node)
			underlying_node_iter = next_underlying_node

		return merged_node_sequence

	@staticmethod
	def construct_underlying_node_sequence_for_merged_node_sequence(merged_node_sequence, underlying_start_node):
		'''merged_node_sequence is a sequence of contiguous descendents in some merged VTPatternNode.
		An underlying_node_sequence is constructed from a contiguous sequence of nodes from the underlying pattern tree
		that contains underlying_start_node. The constructed underlying_node_sequence starts at underlying_start_node.
		The constructed underlying_node_sequence maintains correspondence with the merged_node_sequence, i.e.,
			underlying_node_sequence[i] is in merged_node_sequence[i].corresponding_nodes, with the possibility
		that underlying_node_sequence contains only None elements after a point if the underlying pattern tree does not extend
		as far down as the merged pattern tree along the merged_node_sequence.
		'''

		underlying_node_sequence = []

		merged_node_iter, i = merged_node_sequence[0], 0 #assume sequence of length atleast 1
		underlying_node_iter = underlying_start_node
		while 1:
			assert underlying_node_iter in merged_node_iter.corresponding_nodes
			underlying_node_sequence.append( underlying_node_iter )

			i += 1
			if i == len(merged_node_sequence):
				break

			next_merged_node = merged_node_sequence[i]
			assert next_merged_node in merged_node_iter.children
			assert next_merged_node != None

			next_underlying_node = None
			for candidate_next_underlying_node in next_merged_node.corresponding_nodes:
				#check if candidate_next_underlying_node belongs to the correct underlying pattern tree
				if candidate_next_underlying_node in underlying_node_iter.children:
					next_underlying_node = candidate_next_underlying_node
					break #found one from correct underlying pattern

			if next_underlying_node == None:
				underlying_node_sequence += [None] * (len(merged_node_sequence) - len(underlying_node_sequence)) #pad end of sequence with Nones
				break #out of while loop
			else:
				underlying_node_iter = next_underlying_node
				merged_node_iter = next_merged_node

		return underlying_node_sequence


	###

	def __repr__(self):
		cls = None
		if self.containing_linear_segment != None:
			cls = [vt_node_in_cls.get_func_name() for vt_node_in_cls in self.containing_linear_segment]
		try:
			olsi = [self.children.index(linseg[1]) for linseg in self.originating_linear_segments] #indices in self.children for originating linseg subtrees
		except:
			sys.exit("VTPatternNode.__repr__(): ERROR: first internal node of originating-linear-segment not found among children:"
					"\n  corr_node0 = %s\n  corr_node1 = %s\n  self.originating_linear_segments = %s\n  self.children = %s\n self.ids = %s\n" \
							% ( self.corresponding_nodes[0], self.corresponding_nodes[1] if len(self.corresponding_nodes) > 1 else None,
								[ [(node.func_name, node.get_extract_tree_ids()) for node in ols] for ols in self.originating_linear_segments ],
								common.list_repr_indented(self.children), self.get_extract_tree_ids() )
			)
		result = 'VTPatternNode("%s" %s (%s) "%s" cls=%s olsi=%s) ' \
					% (self.get_func_name(), self.get_extract_tree_ids(), len(self.corresponding_nodes), self.get_node_type(), cls, olsi)
		if VTPatternNode.enable_stats_printing == True:
			result += "%s " % (self.stats, )
		result += common.list_repr_indented(self.children)
		return result



def make_nesting_tuples_of_call_context(call_context):
	if len(call_context) == 0:
		return ()
	return ( call_context[0], make_nesting_tuples_of_call_context(call_context[1:]) )




# Definitions:
#   top-segment till vtr = prefix of a linear-segment from originating task till node vtr, and vtr cannot be the terminating contributing-element
#   bottom-segment from vtr = suffix of linear-segment from vtr till terminating contributing-element, and vtr cannot be the originating task
#
#   linear-segments matching in structure till vtr1 and vtr2 = top-segment(vtr1) and top-segment(vtr2) have corresponding nodes identically named
#     and with sufficiently matching children_call_context_sets.
#

#####################################
### Merging VTPatternNode patterns
#####################################

def get_combined_func_name(vtn1, vtn2):
	'''Atmost one of vtn1, vtn2 may be None.
	Return None if func-names of vtn1 and vtn2 mismatch.
	'''

	if vtn1 != None and vtn2 != None:
		if vtn1.get_func_name() != vtn2.get_func_name():
			return None #mismatch
		return vtn1.get_func_name() #both match

	if vtn1 != None:
		return vtn1.get_func_name()
	else:
		return vtn2.get_func_name()


def get_combined_node_type(vtn1, vtn2):
	'''Atmost one of vtn1, vtn2 may be None.
	Return None if node-types of vtn1 and vtn2 mismatch.
	'''

	if vtn1 != None and vtn2 != None:
		if vtn1.get_node_type() != vtn2.get_node_type():
			return None #mismatch
		return vtn1.get_node_type() #both match

	if vtn1 != None:
		return vtn1.get_node_type()
	else:
		return vtn2.get_node_type()

	
#########################
# Pass 1: merge VTPattern tree structure

def merge_children_call_contexts(mvtn):
	'''Merge children of mvtn.corresponding_nodes.
	Checks whether call-contexts sets are *identical* for merged corresponding children nodes.
	Return True if successful merge.
	Return False if merge conflicts between children-call-contexts-sets
	Precondition: mvtn.corresponding_nodes must be defined
	'''

	map_child_call_context_to_merged_child_node_and_cccset_size = {}

	for corr_location, vtn in enumerate(mvtn.corresponding_nodes):
		if vtn == None:
			continue

		for (c_vtn, cccs_to_c_vtn) in zip(vtn.children, vtn.children_call_context_sets):
			# Check if current children-call-context-sets are compatible with prior children-call-contexts-sets
			#  - all children-call-contexts in current set should map to same merged_node, else all to None
			cccs_merge_node = None
			prior_cccs_size = None
			for i, cc in enumerate(cccs_to_c_vtn):
				tupled_cc = make_nesting_tuples_of_call_context(cc)
				if map_child_call_context_to_merged_child_node_and_cccset_size.has_key(tupled_cc):
					cc_merge_node, prior_cccs_size = map_child_call_context_to_merged_child_node_and_cccset_size[tupled_cc]
				else:
					cc_merge_node = None
				if i > 0 and cccs_merge_node != cc_merge_node:
					return False #child call-context mismatch

				if i == 0: #merge node for call-context-set determined from first call-context encountered in the set
					cccs_merge_node = cc_merge_node

			if cccs_merge_node == None: #call-context not seen before, new child merge-node for mvtn needs to be created
				cccs_merge_node = VTPatternNode()
				cccs_merge_node.corresponding_nodes = [None] * len(mvtn.corresponding_nodes)

				cccs_merge_node.parent = mvtn

				mvtn.children.append(cccs_merge_node)
				mvtn.children_call_context_sets.append(cccs_to_c_vtn)

				prior_cccs_size = len(cccs_to_c_vtn)
				for cc in cccs_to_c_vtn:
					tupled_cc = make_nesting_tuples_of_call_context(cc)
					map_child_call_context_to_merged_child_node_and_cccset_size[tupled_cc] = (cccs_merge_node, prior_cccs_size)

			cccs_merge_node.corresponding_nodes[corr_location] = c_vtn

			if prior_cccs_size != len(cccs_to_c_vtn):
				return False #cccs_to_c_vtn has fewer elements than cccs of a prior vtn for same merge-node

	return True # successful merge

def merge_helper_VTPatternNode_tree_structure(mvtn):
	'''mvtn.corresponding_nodes must be defined'''

	mvtn.func_name = get_combined_func_name(mvtn.corresponding_nodes[0], mvtn.corresponding_nodes[1])
	if mvtn.func_name == None:
		return None #mismatch on func-names

	mvtn.node_type = get_combined_node_type(mvtn.corresponding_nodes[0], mvtn.corresponding_nodes[1])
	if mvtn.node_type == None:
		return None #mismatch on node-types

	#Following creates children merge-nodes, sets 'parent' and 'corresponding_nodes' fields for created merge-nodes,
	#   and updates 'children' and 'children_call_context_sets' fields for mvtn
	if merge_children_call_contexts(mvtn) == False:
		return None #mismatch on merging children call-contexts

	# Recurse
	for c_mvtn in mvtn.children:
		if merge_helper_VTPatternNode_tree_structure(c_mvtn) == None:
			return None

	return mvtn

def merge_VTPatternNode_tree_structure(vtr1, vtr2):
	'''Return mvtr (the merged tree root) if merge succeeds.
	Return None on merge conflict on tree structure.
	For every node mvtn in the tree rooted at mvtr:
		mvtn.corresponding_nodes = [corresponding-node-from-vtr1, corresponding-node-from-vtr2]
		  (a corresponding-node may be None, if that part of the tree does not exist in one of vtr1 or vtr2)
	'''

	mvtr = VTPatternNode() #parent = None
	mvtr.corresponding_nodes = [vtr1, vtr2]
	return merge_helper_VTPatternNode_tree_structure(mvtr)



#########################
# Pass 2: verify merged contribution structure (linear segments)

def verify_merged_VTPatternNode_contribution_structure_for_merged_node(mvtn):
	'''Return True if merge of contribution structure does not conflict between mvtn.corresponding_nodes, False otherwise.
	Assumes that mvtn is a successful merger of the *tree-structure* of the patterns
	in mvtn.corresponding_nodes.
	'''

	# Case Analysis:
	# 1) Suffix verification: Merging patterns at roots (similarly for merged subpatterns at subpattern roots)
	#    a) If patterns have identical structures (node-names and node-types)
	#       - Contribution structure may still not match
	#           (pattern1: T1 -> T2 where T2 contributes to T1, pattern2: T1 -> T2 where T2 does not contribute to T1)
	#    b) If patterns have differing structure, node-types or node-names
	#       - Cannot be in tree-structure conflict (node-names, node-types or call-context-sets), else tree-structure merge would have failed
	#       - Non-conflicting tree-structure:
	#           i. vtn1 has X as a subtree, vtn2 does not have X as a subtree, and
	#                 path from vtn1 to its root matches path from vtn2 to its root (node-names and node-types)
	#              -- Is there a linear segment from vtn1 to X?
	#                   ==> No conflict in contribution structure, but may have ???Interpretation Implications??? for overall merged structure
	#
	#          ii. Both vtn1 and vtn2 have subtree root X as child, although the subtrees may differ under X for the two patterns
	#              -- if neither vtn1 nor vtn2 have a linear segment extending over them to X
	#                   ==> No conflict
	#              -- If only one of vtn1 or vtn2 has a linear segment extending over to X
	#                   ==> conflict
	#              -- If both vtn1 and vtn2 have linear segments extending over to X
	#                   + If both linear segments are identical in their entirety ==> No conflict
	#                   + If the linear segments differ even in their prefix before vtn1 or vtn2 or in their suffix after X ==> conflict
	#
	# 2) Prefix verification: Merging subpatterns vtn1 and vtn2, which are children of a single parent pattern, with P the immediate parent node of both
	#    a) If no linear segment extends from P to either vtn1 or vtn2
	#        ==> No conflict in contribution structure implied
	#    b) If a linear segment extends across P and vtn1 but there is no other linear segment across P and vtn2
	#        ==> conflict
	#    c) If a linear segment L1 extends across P to vtn1 and a linear segment L2 extends across P to vtn2:
	#        Implication: P has to be the originating task of both L1 and L2, as a two linear segments cannot be overlapping
	#        - If L1 and L2 are both identical paths, even though vtn1 and vtn2 may not be identical in their subtrees
	#          ==> No conflict
	#        - o.w.
	#          ==> conflict


	# Premise for Algorithm:
	#   - Ignore nature of nodes, just look at the tree structure imposed on the pattern by the contribution structure
	#   - Therefore, focus on defining acceptable merges between the overlayed contribution structure
	#
	# Algorithm
	# Step #1) At current merge node mvtn, in some traversal (say, pre-order) of the merged pattern tree-structure
	#   a) if any corresponding node vtn_i of mvtn is part of a linear segment
	#      then the following must hold for any other corresponding node vtn_j, otherwise there is conflict in contribution structure
	#     - if vtn_i is the originator for a linear segment L, and vtn_j lacks the entire subtree that could contain L
	#         ++ then, vtn_j does not imply a conflict on L (but still could on another linear segment L2 originating at vtn_i)
	#     - o.w., vtn_j must be part of an identical linear segment in its corresponding pattern tree, with vtn_i and vtn_j
	#          in the same position in their respective linear segments
	#
	# Step #2) If any vtn_i is the originator for a linear segment, and no conflicts were detected at mvtn for any linear segment,
	#    then construct the contribution structure information for mvtn
	#     (update the 'originating_linear_segments' and 'containing_linear_segment' fields for affected merged nodes in the mvtn merged subtree)

	#Step #1
	for vtn_i in [corr_node for corr_node in mvtn.corresponding_nodes if corr_node != None]:
		#check on originating linear segments
		for linseg_i in vtn_i.originating_linear_segments: #zero or more linear segments may originate on vtn_i
			# Now at least one linear segment originates at vtn_i

			assert len(linseg_i) >= 2

			merged_sequence_for_linseg_i = mvtn.construct_merged_node_sequence_for_underlying_node_sequence(linseg_i)
			for vtn_j in [corr_node for corr_node in mvtn.corresponding_nodes if corr_node != None and corr_node != vtn_i]:
				potential_linseg_j = VTPatternNode.construct_underlying_node_sequence_for_merged_node_sequence(merged_sequence_for_linseg_i, vtn_j)
				if potential_linseg_j[1] != None:
					#subtree that could contain corresponding linear segment for linseg_i exists below vtn_j
					if potential_linseg_j not in vtn_j.originating_linear_segments:
						return False #no corresponding linear segment found originating at vtn_j

		#check on containing linear segment
		if vtn_i.containing_linear_segment != None:
			linseg_i = vtn_i.containing_linear_segment
			assert len(linseg_i) >= 2
			originating_node_for_linseg_i = linseg_i[0]
			originating_node_for_merged_sequence_for_linseg_i = mvtn.get_merged_ancestor_node_corresponding_to_underlying_ancestor_node(originating_node_for_linseg_i)
			merged_sequence_for_linseg_i = originating_node_for_merged_sequence_for_linseg_i.construct_merged_node_sequence_for_underlying_node_sequence(linseg_i)

			for vtn_j in [corr_node for corr_node in mvtn.corresponding_nodes if corr_node != None and corr_node != vtn_i]:
				linseg_j = vtn_j.containing_linear_segment
				if linseg_j == None:
					return False
				assert len(linseg_j) >= 2
				originating_node_for_linseg_j = linseg_j[0]
				originating_node_for_merged_sequence_for_linseg_j = mvtn.get_merged_ancestor_node_corresponding_to_underlying_ancestor_node(originating_node_for_linseg_j)
				merged_sequence_for_linseg_j = originating_node_for_merged_sequence_for_linseg_j.construct_merged_node_sequence_for_underlying_node_sequence(linseg_j)

				if merged_sequence_for_linseg_i != merged_sequence_for_linseg_j:
					return False
	# Now, no merge conflicts on contribution structure detected at mvtn

	#Step #2
	assert len(mvtn.originating_linear_segments) == 0
	#assert mvtn.containing_linear_segment == None #Shouldn't be asserting this as processing a merged parent originator would attach the linear segment for descendent nodes

	for vtn_i in [corr_node for corr_node in mvtn.corresponding_nodes if corr_node != None]:
		#build originating linear segments for merged node
		for linseg_i in vtn_i.originating_linear_segments:
			merged_sequence_for_linseg_i = mvtn.construct_merged_node_sequence_for_underlying_node_sequence(linseg_i)

			assert None not in merged_sequence_for_linseg_i
			assert len(merged_sequence_for_linseg_i) >= 2

			if merged_sequence_for_linseg_i in mvtn.originating_linear_segments:
				continue #already added (should have been added by a corresponding node prior to vtn_i, but we don't explicitly verify that)

			mvtn.originating_linear_segments.append( merged_sequence_for_linseg_i )
			for desc_mvtn in merged_sequence_for_linseg_i[1:]:
				assert desc_mvtn.containing_linear_segment == None, "Failure for mvtn = %s" % (mvtn, )
				desc_mvtn.containing_linear_segment = merged_sequence_for_linseg_i

	return True #no conflicts on contribution structure, constructed contribution structure on mvtn subtree

def verify_merged_VTPatternNode_contribution_structure_recursive(mvtp):
	'''Verify that the contribution structure of the merged-tree rooted at mvtp are not in conflict, and construct merged contribution structure information
	Return True if no contribution structure conflict detected in mvtp merged-tree, o.w. False
	'''

	# Verify in pre-order
	if verify_merged_VTPatternNode_contribution_structure_for_merged_node(mvtp) == False:
		return False

	# Recurse
	for c_mvtp in mvtp.children:
		if verify_merged_VTPatternNode_contribution_structure_recursive(c_mvtp) == False:
			return False

	return True

#########################
# Pass 3: determine merge cost

from varanalysis import get_merged_mean_cov

def update_chain_of_stats(vtp):
	under_vtp_with_stats = vtp
	while under_vtp_with_stats.stats == None:
		assert len(under_vtp_with_stats.corresponding_nodes) == 1
		under_vtp_with_stats = under_vtp_with_stats.corresponding_nodes[0]

	if vtp != under_vtp_with_stats:
		vtp.stats = under_vtp_with_stats.stats.clone()

def compute_and_annotate_node_merge_cost(mvtn):
	actual_corresponding_nodes = [vtp for vtp in mvtn.corresponding_nodes if vtp != None]
	if len(actual_corresponding_nodes) < 2:
		assert len(actual_corresponding_nodes) == 1
		vtp1 = actual_corresponding_nodes[0]
		update_chain_of_stats(vtp1)

		mvtn.stats = vtp1.stats.clone()
		return 0.0 #no cost

	assert len(actual_corresponding_nodes) == 2
	vtp1 = actual_corresponding_nodes[0]
	vtp2 = actual_corresponding_nodes[1]

	update_chain_of_stats(vtp1)
	update_chain_of_stats(vtp2)

	assert vtp1.get_node_type() == vtp2.get_node_type()
	if vtp1.get_node_type() == "task":
		node_D_metric = distribution.kolmogorov_smirnov_difference(vtp1.stats.exec_time_distr, vtp2.stats.exec_time_distr)
		assert 0.0 <= node_D_metric and node_D_metric <= 1.0
	else:
		node_D_metric = 0.0

	num_factors = 1
	
	mvtn.stats = NodeStatistics(
			(vtp1.stats.total_count + vtp2.stats.total_count),
			distribution.Distribution( vtp1.stats.exec_time_distr.mean_std_count_list + vtp2.stats.exec_time_distr.mean_std_count_list ) )

	(mvtn_mean, mvtn_std, mvtn_count) = mvtn.stats.exec_time_distr.get_combined_mean_std_count()
	(vtp1_mean, vtp1_std, vtp1_count) = vtp1.stats.exec_time_distr.get_combined_mean_std_count()
	(vtp2_mean, vtp2_std, vtp2_count) = vtp2.stats.exec_time_distr.get_combined_mean_std_count()


	#Create contribution_stats for merged structure
	for m_i, m_linseg in enumerate(mvtn.originating_linear_segments):
		#1. determine the index of m_linseg in vtp1.originating_linear_segments and vtp2.originating_linear_segments
		#2. combine information from those indices into vtp1.stats.contribution_stats and vtp2.contribution_stats


		corresponding_vtp1_i = None
		for vtp1_i, vtp1_linseg in enumerate(vtp1.originating_linear_segments):
			if vtp1_linseg[-1] in m_linseg[-1].corresponding_nodes:
				corresponding_vtp1_i = vtp1_i
				break

		corresponding_vtp2_i = None
		for vtp2_i, vtp2_linseg in enumerate(vtp2.originating_linear_segments):
			if vtp2_linseg[-1] in m_linseg[-1].corresponding_nodes:
				corresponding_vtp2_i = vtp2_i
				break

		underlying_contrib_stats_for_m_linseg = []
		if corresponding_vtp1_i != None:
			underlying_contrib_stats_for_m_linseg.append( vtp1.stats.contribution_stats[corresponding_vtp1_i] )
		else:
			underlying_contrib_stats_for_m_linseg.append( ContribStatistics( distribution.Distribution( [(0.0, 0.0, vtp1_count)] ), 0.0, 0.0 ) )

		if corresponding_vtp2_i != None:
			underlying_contrib_stats_for_m_linseg.append( vtp2.stats.contribution_stats[corresponding_vtp2_i] )
		else:
			underlying_contrib_stats_for_m_linseg.append( ContribStatistics( distribution.Distribution( [(0.0, 0.0, vtp2_count)] ), 0.0, 0.0 ) )

		underlying_counts = [ucs.exec_time_distr_wrt_variant_task.get_combined_mean_std_count()[2] for ucs in underlying_contrib_stats_for_m_linseg]
		NEW_count = sum(underlying_counts)
		assert NEW_count == mvtn_count

		weights = [count/float(NEW_count) for count in underlying_counts]

		NEW_weighted_sum_of_underlying_contrib_fractions = \
				sum( [w * ucs.weighted_sum_of_underlying_contrib_fractions for w, ucs in zip(weights, underlying_contrib_stats_for_m_linseg)] )
		NEW_weighted_squared_sum_of_underlying_contrib_fractions = \
				sum( [w * ucs.weighted_squared_sum_of_underlying_contrib_fractions for w, ucs in zip(weights, underlying_contrib_stats_for_m_linseg)] )

		NEW_mean_std_count_list_wrt_variant_task = []
		for ucs in underlying_contrib_stats_for_m_linseg:
			NEW_mean_std_count_list_wrt_variant_task += ucs.exec_time_distr_wrt_variant_task.mean_std_count_list

		mvtn.stats.add_contributor_info(m_i,
				distribution.Distribution(NEW_mean_std_count_list_wrt_variant_task),
				NEW_weighted_sum_of_underlying_contrib_fractions, NEW_weighted_squared_sum_of_underlying_contrib_fractions)

	#Compute average spread in contrib_fraction due to merging (all the way from underlying contributors at the lowest level)
	avg_contrib_frac_variance = 0.0
	if mvtn.get_node_type() == "task" and len(mvtn.stats.contribution_stats) != 0:
		contrib_fraction_variance_sum = 0.0
		for i, cs_i in enumerate(mvtn.stats.contribution_stats):
			NEW_contrib_fraction_i = mvtn.stats.get_contrib_fraction(i)
			variance_wrt_NEW_contrib_fraction_i = \
					cs_i.weighted_squared_sum_of_underlying_contrib_fractions \
					- 2.0 * NEW_contrib_fraction_i * cs_i.weighted_sum_of_underlying_contrib_fractions \
					+ NEW_contrib_fraction_i * NEW_contrib_fraction_i
			contrib_fraction_variance_sum += variance_wrt_NEW_contrib_fraction_i

		avg_contrib_frac_variance = contrib_fraction_variance_sum / float( len(mvtn.stats.contribution_stats) )
		num_factors += 1

	node_merge_cost = (node_D_metric + avg_contrib_frac_variance) / float(num_factors)

	return node_merge_cost


merge_cost_limit = 0.8 #-1.0
if setting_merge_cost_limit != None:
	merge_cost_limit = setting_merge_cost_limit
print "~~~ USING merge_cost_limit = %s for MERGING ~~~" % (merge_cost_limit, )

def progressive_bfs_compute_merge_cost(mvtp):
	Q = [mvtp]
	progressive_merge_cost = 0.0
	nodes_processed = 0

	#process queue
	while len(Q) > 0:
		mn = Q[0]
		Q[:1] = [] #dequeue head
		Q.extend( [c for c in mn.children if c != None] ) #add children at end

		progressive_merge_cost = (progressive_merge_cost * nodes_processed + compute_and_annotate_node_merge_cost(mn)) / (nodes_processed + 1.0)
		nodes_processed += 1

		if progressive_merge_cost > merge_cost_limit:
			return progressive_merge_cost #terminate, progressive cost so far has exceeded limit

	if not (0 <= progressive_merge_cost and progressive_merge_cost <= merge_cost_limit):
		sys.exit("progressive_merge_cost(): ERROR: progressive_merge_cost = %s\n" % (progressive_merge_cost,))
	return progressive_merge_cost #less than cost limit
	

#####################################
### Multi-layer merge driver
#####################################

vt_pattern_layer0 = []
for etr in extracted_trees:
	vt_pattern_layer0.append( VTPatternNode(etr).construct_linear_segments() )

print "### NO INTERNAL MERGES vt_pattern_layer0 = "
print common.list_repr_indented(vt_pattern_layer0)



def merge_layer_of_VTPatternNode(previous_layer): #FIXME: TO DOs: need to accomodate a "merge-pressure parameter"
	'''Produce a new layer, next_layer, of merged patterns. A merged pattern in next_layer will
	have corresponding nodes in patterns from previous_layer. If a pattern in previous_layer was
	not merged, it will directly be part of next_layer.

	Return (next_layer, merged_pattern_indices), where merged_pattern_indices is a list with elements corresponding to
		elements of next_layer.
		The k-th element of merged_pattern_indices is a list of indices into previous_layer corresponding to patterns
		that got merged to produce the k-th pattern in next_layer. If a pattern in previous_layer was not merged,
		the same pattern object will occur in next_layer and its corresponding entry in merged_pattern_indices will be a list
		of length == 1. Therefore, an element of length == 1 in merged_pattern_indices is the test for identifying patterns
		from previous_layer that were not merged into a new pattern.
	'''

	### Algo
	# 1. Combine every pair of patterns in previous_layer that merge without structural (tree or contribution) conflicts
	#      - "merge-cost" determined for each merge pair
	# 2. Sort merged pairs by merge-cost
	# 3. In ascending order of merge-costs for merge-pairs (k = (i, j), merge-cost), where k is the merged pattern of i and j
	#      - if neither i nor j were part of an already "Accepted" merge-pair,
	#           ++ Accept k = (i, j): add k to next_layer, mark each of i and j as "consumed"
	# 4. For every i in previous_layer
	#      - if i is not marked "consumed"
	#           ++ add i to next_layer

	merged_pair_candidates = []
		# Each list element is a tuple of form: (i, j, merged_new_vtp, merge_cost), where i, j are indices into previous_layer

	# Merge every possible pair
	for i, vtp_i in enumerate(previous_layer):
		for j, vtp_j in enumerate(previous_layer):
			if i == j:
				continue

			# Merge Pass 1
			mvtp = merge_VTPatternNode_tree_structure(vtp_i, vtp_j)

			# Merge Pass 2
			if mvtp != None:
				print "Verifying Contribution Structure of merged pattern = ", mvtp
			if mvtp != None and verify_merged_VTPatternNode_contribution_structure_recursive(mvtp) == False:
				# merger of contribution structure failed
				mvtp = None

			if mvtp != None:
				merge_cost = progressive_bfs_compute_merge_cost(mvtp)
			else:
				merge_cost = 0.0

			if merge_cost > merge_cost_limit:
				mvtp = None #prevent merging

			if mvtp != None:
				print "Hooray! (%s, %s) tree and contrib structures merge with cost = %s" % (i, j, merge_cost)
				merged_pair_candidates.append( (i, j, mvtp, merge_cost) )


	#sort candidates by ascending order of merge-cost
	merged_pair_candidates.sort( lambda x, y: cmp(x[3], y[3]) )

	#Promote merged candidates to next_layer
	next_layer = []
	merged_pattern_indices = []

	previous_layer_pattern_indices_consumed_status = [False] * len(previous_layer) #True => consumed
	for (i, j, mvtp, merge_cost) in merged_pair_candidates:
		if previous_layer_pattern_indices_consumed_status[i] == True or previous_layer_pattern_indices_consumed_status[j] == True:
			continue

		# neither i nor j have been consumed so far
		#FIXME: Insert logic for trading-off "merge-pressure" against "merge_cost" here, and make addition to next_layer conditional
		next_layer.append( mvtp )
		merged_pattern_indices.append( [i, j] )

		previous_layer_pattern_indices_consumed_status[i] = True
		previous_layer_pattern_indices_consumed_status[j] = True

	#Promote any previous_layer patterns that were not merged
	for i, vtp_i in enumerate(previous_layer):
		if previous_layer_pattern_indices_consumed_status[i] == False:
			next_layer.append(vtp_i)
			merged_pattern_indices.append( [i] )

	return (next_layer, merged_pattern_indices)

def merge_layers_of_VTPatternNode_until_stabilized(initial_layer, name_to_print = None):
	'''Return tuple (merge_layers, final_merge_layer_merged_indices),
		where merge_layers is list of merged layers, the first element being the initial_layer
		and final_merge_layer_merged_indices is a list of elements corresponding to elements of merge_layers[-1] (i.e. to the final layer).

		The k-th element of final_merge_layer_merged_indices gives the indices of the patterns in initial_layer that
		were merged to produce the k-th pattern in the final layer of merge_layers.
	'''

	merge_layers = [initial_layer]
	final_merge_layer_merged_indices = None #this should not be accessed if len(merge_layers) == 1

	previous_final_merge_layer_merged_indices = [[i] for i in range( len(initial_layer) )]
		#each element of initial_layer corresponds to merging with just itself

	layer_num = 0
	previous_layer = initial_layer
	while 1:
		(next_layer, merged_pattern_indices) = merge_layer_of_VTPatternNode(previous_layer)
		layer_num += 1

		if len(next_layer) == len(previous_layer):
			break # stabilized

		merge_layers.append( next_layer )
		final_merge_layer_merged_indices = []
		for k, mpi in enumerate(merged_pattern_indices):
			new_mpi = []
			for prev_index in mpi:
				new_mpi.extend( previous_final_merge_layer_merged_indices[prev_index] )
			final_merge_layer_merged_indices.append( new_mpi )
		assert len(final_merge_layer_merged_indices) == len(next_layer)

		if name_to_print != None:
			print "### %s[%s] = " % (name_to_print, layer_num)
			print common.list_repr_indented( zip(next_layer, final_merge_layer_merged_indices) )

		previous_layer = next_layer
		previous_final_merge_layer_merged_indices = final_merge_layer_merged_indices
	return (merge_layers, final_merge_layer_merged_indices)



#########################
# Internal Merging in a Pattern

def incrementally_fix_contribution_structure_due_to_replicated_children(vtn, ret_vtn):
	'''vtn is read-only. ret_vtn should be the VTPatternNode allocated to replace vtn.
	ret_vtn.children must be the *replicated* children subtrees for vtn.children, not the original children subtrees of vtn.
	This function fixes the contribution structure spanning ret_vtn, its *unreplicated* ancestors and its *replicated* children,
	by possibly updating the 'containing_linear_segment' fields of ret_vtn and its immediate children,
	and the 'originating_linear_segments' field of ret_vtn. All updates to 'containing_linear_segment' would result in
	incrementally/partially updated linear segment information (mixing unreplicated ancestors and replicated nodes in ret_vtn subtree),
	whereas the update to 'originating_linear_segments' field would completely update the linear segments to be constructed wholly from
	replicated nodes under ret_vtn.
	'''

	#Incrementally Patch-up Contribution Structure that spans from vtn or above into replicated subtrees
	# Algo: *Partially* replicate or update linear segments (to deal with replication of subtrees)
	#    (Tricky!! effect must be contained within top-most recursive invocations of help_merge_internal_hierarchy_of_VTPatternNode_recursive(),
	#         since VTPatternNode data structure is not aware of linear segments that mix underlying and merged pattern nodes)
	#     - If vtn is a contributor of variance along linear segment L_underlying, then create new linear segment object L_new s.t.
	#         L_new[-1] = ret_vtn (i.e. new replicated node)
	#         L_new[0..len-1] = L_underlying[0..len-1]
	#         Set ret_vtn.containing_linear_segment = L_new
	#
	#     - If vtn is internal to a linear segment L_underlying, then determine corresponding L_new from correct replicated child node
	#             (there can be only one such child node), and modify L_new[position of vtn in L_underlying] = ret_vtn
	#         Set ret_vtn.containing_linear_segment = L_new
	#
	#     - If vtn is the originator of one of more linear segments, then iterate through ret_vtn.children, and for each child rc:
	#         if rc.containing_linear_segment != None, then fix rc.containing_linear_segment[0] = ret_vtn (should have vtn before updating)
	#          and append the updated rc.containing_linear_segment to ret_vtn.originating_linear_segments

	# ret_vtn originator of linear segments
	assert ret_vtn.originating_linear_segments == []
	for rc in ret_vtn.children:
		if rc.containing_linear_segment != None: #the linear segment should have been partially fixed up for replication by lower-level replication recursions
			if rc.containing_linear_segment[0] == vtn: #vtn was originator in partially updated linear segment
				rc.containing_linear_segment[0] = ret_vtn
				ret_vtn.originating_linear_segments.append( rc.containing_linear_segment )

	# ret_vtn contributor or internal to a linear segment
	if vtn.containing_linear_segment != None:
		if vtn.containing_linear_segment[-1] == vtn: #vtn is a contributor
			L_new = vtn.containing_linear_segment[:] #copied list object, vtn is unmodified
			L_new[-1] = ret_vtn
		else: #vtn is internal
			loc = vtn.containing_linear_segment.index(vtn)
			assert loc > 0 and loc < len(vtn.containing_linear_segment)

			# Now, exactly one child node will be contained in a linear segment. Find it.
			linseg_child_loc = -1
			for ci, c in enumerate(ret_vtn.children):
				if c.containing_linear_segment != None:
					assert vtn in c.containing_linear_segment
					assert linseg_child_loc == -1 #there should be exactly one found
					linseg_child_loc = ci
			assert linseg_child_loc != -1
			
			L_new = ret_vtn.children[linseg_child_loc].containing_linear_segment
				#same list object will be updated (used by all elements in this linear segment below vtn)
			assert L_new[loc] == vtn
			L_new[loc] = ret_vtn

		assert ret_vtn.containing_linear_segment == None
		ret_vtn.containing_linear_segment = L_new

def help_merge_internal_hierarchy_of_VTPatternNode_recursive(vtn, retry_merging_on_merged_subpatterns_flag, force_replication_flag = False):
	'''Attempt to merge sibling nodes bottom-up in subtree rooted at vtn.
	force_replication_flag = True forces subtree to be replicated even if no merging occurs anywhere in subtree rooted at vtn.
	force_replication_flag = False will not replicate subtree if no merging occurs.

	Return (replication_occured_flag = True/False, replicated and possibly merged tree for vtn).
	replication_occured_flag == False implies that no merging occured and replication was not explicitly requested.
	Always returns fully replicated and suitably merged sub-tree corresponding to vtn regardless of whether merging occured anywhere.
	Warning: the contribution structure of replicated tree may only get partially updated if this function is not invoked at highest level of a pattern

	Assumption: All linear segment completely contained within the vtn subtree must be *complete* linear segment.
	     However, the linear segment incoming to vtn (if any) could be *partial* or *complete*.
		 If the incoming linear segment in partial, then it must satisfy the following:
		 - the suffix of the linear segment consists of nodes in the vtn subtree
		 - the prefix consists of ancestors from an underlying pattern (not necessarily immediately underlying)
	Note: The incoming linear segment is allowed to be partial because this algorithm never accesses vtn.parent.children.

	Note: data-structure subtree rooted at vtn is treated as read-only
	'''

	# Setup replicated version of vtn just in case it is needed
	ret_vtn = VTPatternNode() #replacement object for vtn or throw-away object
	ret_vtn.corresponding_nodes.append( vtn )
	ret_vtn.func_name = vtn.func_name
	ret_vtn.node_type = vtn.node_type
	ret_vtn.parent = None


	####
	# Phase I: give merging a chance to occur within children subtrees
	flags_and_merged_children_subtrees = [retry_help_merge_internal_hierarchy_of_VTPatternNode_recursive(c_vtn, retry_merging_on_merged_subpatterns_flag, force_replication_flag) for c_vtn in vtn.children]

	flags = [rep_flag for (rep_flag, merging_of_immediate_children_occured_in_c_vtn, rep_c_vtn) in flags_and_merged_children_subtrees]
	merged_children_subtrees = [rep_c_vtn for (rep_flag, merging_of_immediate_children_occured_in_c_vtn, rep_c_vtn) in flags_and_merged_children_subtrees]

	merging_occured_within_any_child = (flags.count(True) > 0)
	if merging_occured_within_any_child == True:
		force_replication_flag = True #turn replication on globally for vtn (if it was not already on)

	ret_vtn.children = merged_children_subtrees
	for rc in ret_vtn.children:
		rc.parent = ret_vtn
	ret_vtn.children_call_context_sets = vtn.children_call_context_sets[:] #make copy

	#fix up contribution structure below and upto vtn (containing_linear_segment fields in ret_vtn and children subtrees, and originating_linear_segments in ret_vtn)
	incrementally_fix_contribution_structure_due_to_replicated_children(vtn, ret_vtn)

	# Now, elements of ret_vtn.children are replicated (possibly with internal mergers) subtrees corresponding to elements of vtn.children.
	#  The contribution structure in ret_vtn is completely replicated, and the linear segment incoming to ret_vtn would be a partial linear segement.


	####
	# Phase II: Attempt merger between immediate children of ret_vtn, with suitable scaffolding added if vtn was in the proper suffix of a linear segment

	#Add scaffolding
	if ret_vtn.containing_linear_segment != None and ret_vtn.containing_linear_segment[-1] != ret_vtn:
	#scaffolding needed if ret_vtn is contained in but does not terminate a linear segment
		loc = ret_vtn.containing_linear_segment.index(ret_vtn)
		assert loc > 0 and loc < len(ret_vtn.containing_linear_segment) - 1
		assert vtn.containing_linear_segment[loc] == vtn

		#create linear scaffolding of VTPatternNode elements upto vtn
		scaffolding_linseg_prefix = []
		for i, vtn_linseg_iter in enumerate( vtn.containing_linear_segment[:loc] ): #till just before loc
			scaffolding_iter = VTPatternNode()
			scaffolding_iter.corresponding_nodes.append( vtn_linseg_iter )
			scaffolding_iter.func_name = "scaffolding_" + vtn_linseg_iter.func_name
			scaffolding_iter.node_type = vtn_linseg_iter.node_type
			if i == 0:
				scaffolding_iter.parent = None #structure above start of linear segment is not needed

				scaffolding_iter.originating_linear_segments.append( ret_vtn.containing_linear_segment )
			else:
				scaffolding_iter.parent = scaffolding_linseg_prefix[-1]
				scaffolding_linseg_prefix[-1].children.append( scaffolding_iter )

				scaffolding_iter.containing_linear_segment = ret_vtn.containing_linear_segment

			scaffolding_linseg_prefix.append( scaffolding_iter )

		#attach scaffolding, fix up ret_vtn.containing_linear_segment linear segment object
		assert ret_vtn.parent == None
		ret_vtn.parent = scaffolding_linseg_prefix[-1]
		scaffolding_linseg_prefix[-1].children.append( ret_vtn )

		ret_vtn.containing_linear_segment[:loc] = scaffolding_linseg_prefix #partial linear segment object is now temporarily modified to a complete linear segment over scaffolding

	#Add proxies
	if len(ret_vtn.originating_linear_segments) > 0: #there are originating linear segments
		assert ret_vtn.parent == None #No scaffolding should have been added. The use of proxies and scaffolding should occur in non-overlapping circumstances

		patterns_to_merge = []
		for i, rc in enumerate(ret_vtn.children):
			proxy_ret_vtn = VTPatternNode()
			proxy_ret_vtn.corresponding_nodes.append( vtn ) #Important to correspond to vtn and not ret_vtn, as proxies replace ret_vtn and not add another layer
			proxy_ret_vtn.func_name = "proxy_" + ret_vtn.func_name
			proxy_ret_vtn.node_type = ret_vtn.node_type

			proxy_ret_vtn.parent = None #structure above vtn not needed

			#Detach and Acquire children of ret_vtn
			rc.parent = proxy_ret_vtn
			proxy_ret_vtn.children.append( rc )

			forced_identical_call_context = [("proxy_cc", 0)]
				#force an *identical* call-context to child so that tree-merging cannot occur unless child node matches up with child node of other proxy pattern
			proxy_ret_vtn.children_call_context_sets.append( forced_identical_call_context )
			#FIXME: error emanates from here, probably

			proxy_ret_vtn.containing_linear_segment = None #since contribution structure does not traverse ret_vtn in this case

			if rc.containing_linear_segment != None:
				assert rc.containing_linear_segment[0] == ret_vtn #must be originating at ret_vtn, since traversing contribution structure case is excluded
				rc.containing_linear_segment[0] = proxy_ret_vtn
				proxy_ret_vtn.originating_linear_segments.append( rc.containing_linear_segment )

			patterns_to_merge.append( proxy_ret_vtn )

		proxies_added = True

	else: #there are no linear segments originating at ret_vtn
		patterns_to_merge = ret_vtn.children
		proxies_added = False


	# Now, tree structures examined for merging will have well-defined contribution structure, including for an incoming linear segment from parents
	### Attempt Merging of immediate children of ret_vtn
	print "patterns_to_merge = ", common.list_repr_indented(patterns_to_merge)
	merged_children_layers, final_merge_layer_merged_indices = merge_layers_of_VTPatternNode_until_stabilized(patterns_to_merge)
	assert len(merged_children_layers) >= 1

	#Detach scaffolding, revert to partial linear segment
	if ret_vtn.parent != None: #has scaffolding
		assert proxies_added == False #addition of scaffolding and proxies occurs in mutually exclusive cases

		loc = ret_vtn.containing_linear_segment.index(ret_vtn)
		assert loc > 0
		assert vtn.containing_linear_segment[loc] == vtn

		#convert back to a partial linear segment
		ret_vtn.containing_linear_segment[:loc] = vtn.containing_linear_segment[0:loc]

		#detach
		assert ret_vtn.parent.func_name[0:len("scaffolding_")] == "scaffolding_"
		ret_vtn.parent.children[:] = []
		ret_vtn.parent = None

	#Remove proxies
	if proxies_added == True:
		print "Just before removing proxies: ret_vtn =", ret_vtn
		print "merged_children_layers =", common.list_repr_indented(merged_children_layers)
		#Remove proxies from unmerged children of ret_vtn
		for rc in ret_vtn.children:
			proxy = rc.parent
			assert proxy.func_name[0:len("proxy_")] == "proxy_"
			assert len(proxy.children) == 1
			actual_ret_vtn_child = proxy.children[0]
			assert rc == actual_ret_vtn_child
			#proxy.children untouched to allow trimming of proxies from final merge layer to work (since some initial layer patterns may propagate unmerged to final layer)

			actual_ret_vtn_child.parent = ret_vtn
			if actual_ret_vtn_child.containing_linear_segment != None:
				assert actual_ret_vtn_child.containing_linear_segment[0] == proxy
				actual_ret_vtn_child.containing_linear_segment[0] = ret_vtn

		#Eliminate proxies from final merge layer patterns
		proxy_removed_final_layer = []
		for mp in merged_children_layers[-1]:
			assert mp.func_name[0:len("proxy_")] == "proxy_"
			assert len(mp.children) == 1
			actual_child_subtree_root = mp.children[0]

			actual_child_subtree_root.parent = ret_vtn #if this layer gets used, then make sure the linear segments of patterns originate at ret_vtn instead of proxies
			if actual_child_subtree_root.containing_linear_segment != None:
				assert actual_child_subtree_root.containing_linear_segment[0] == mp or actual_child_subtree_root.containing_linear_segment[0] == ret_vtn
				actual_child_subtree_root.containing_linear_segment[0] = ret_vtn

			proxy_removed_final_layer.append( actual_child_subtree_root )
		merged_children_layers[-1] = proxy_removed_final_layer #replace final layer with trimmed patterns
		print "AFTER removing proxies: merged_children_layers = ", merged_children_layers

	# Now, any added scaffolding or proxies have been removed.

	merging_occured_at_ret_vtn = len(merged_children_layers) >= 2
	if merging_occured_at_ret_vtn:
		force_replication_flag = True #turn replication on globally for vtn

		updated_children = merged_children_layers[-1]
		updated_children_merge_indices = final_merge_layer_merged_indices

		before_merging_children = ret_vtn.children
		ret_vtn.children = updated_children
		for uc in ret_vtn.children:
			uc.parent = ret_vtn

		#Setup ret_vtn.children_call_context_sets from vtn.children_call_context_sets (which are still valid after replication of vtn.children, but prior to merging)
		ret_vtn.children_call_context_sets[:] = [[] for c in ret_vtn.children] #create correct number of empty lists
		for k, ucmi_k in enumerate(updated_children_merge_indices):
			for underlying_index in ucmi_k: #indices into vtn.children that were merged to form updated_children[k]
				ret_vtn.children_call_context_sets[k].extend( vtn.children_call_context_sets[underlying_index] )

		#Merging Contribution Structure at originating node:
		# If merging across children occured, then the following would hold separately for any group of merged children
		# Either:
		#   i) None of the children in a merge-group are contained inside linear segments
		#    OR
		#   ii) The merge-group consists of one child and that child is contained inside a linear segment that originates at an ancestor of vtn
		#    OR
		#   iii) All of the children in a merge-group are contained inside identical linear segments, all of which originate at vtn
		#
		#  These conditions do not preclude children in any merged group from themselves originating linear segments in their sub-trees.
		#
		# ASSERT on these conditions, just in case

		ret_vtn.originating_linear_segments[:] = []
		print "updated_children_merge_indices = ", updated_children_merge_indices
		print "updated_children = ", common.list_repr_indented(updated_children)
		print "before_merging_children = ", common.list_repr_indented(before_merging_children)
		for k, ucmi_k in enumerate(updated_children_merge_indices):
			assert len(ucmi_k) >= 1
			if len(ucmi_k) == 1:
				#no merging layers were added for this subpattern

				only_underlying_index = ucmi_k[0]
				assert before_merging_children[only_underlying_index] == ret_vtn.children[k] #must be original, since no merging occured
				only_containing_linear_segment = ret_vtn.children[k].containing_linear_segment

				if only_containing_linear_segment != None:
					assert vtn not in only_containing_linear_segment #incremental update on partial linear segment should have been carried out prior to merging attempt
					if only_containing_linear_segment[0] == ret_vtn: #need to re-insert
						ret_vtn.originating_linear_segments.append( only_containing_linear_segment )
					#otherwise: originator is an ancestor of vtn, nothing to do w.r.t. ret_vtn.originating_linear_segments

			else: #multiple children in k'th merge-group
				#verify
				num_orig_children_contained_in_linsegs_for_merged_group_k = 0 #count how many children are contained in linear segments originating at ret_vtn
				for underlying_index in ucmi_k:
					if before_merging_children[underlying_index].containing_linear_segment != None:
						assert before_merging_children[underlying_index].containing_linear_segment[0] == ret_vtn
						num_orig_children_contained_in_linsegs_for_merged_group_k += 1
				print "ASSERT: k = %s ucmi_k = %s" % (k, ucmi_k)
				print "ret_vtn = ", ret_vtn
				print "num_orig_children_contained_in_linsegs_for_merged_group_k = %s, ucmi_k = %s" % (num_orig_children_contained_in_linsegs_for_merged_group_k, ucmi_k)
				assert num_orig_children_contained_in_linsegs_for_merged_group_k == 0 \
						or num_orig_children_contained_in_linsegs_for_merged_group_k == len(ucmi_k)

				#insert merged version of linear segment
				if num_orig_children_contained_in_linsegs_for_merged_group_k > 0: #need to explicitly reconstruct merged containing_linear_segment for merge-group
					underlying_linseg = ret_vtn.children[k].containing_linear_segment

					assert underlying_linseg[0] == ret_vtn
					ret_vtn.originating_linear_segments.append( underlying_linseg )

	update_chain_of_stats(ret_vtn)

	return (force_replication_flag, merging_occured_at_ret_vtn, ret_vtn)


#Note: MUTUAL RECURSION between retry_help_merge_internal_hierarchy_of_VTPatternNode_recursive() and help_merge_internal_hierarchy_of_VTPatternNode_recursive()
def retry_help_merge_internal_hierarchy_of_VTPatternNode_recursive(vtn, retry_merging_on_merged_subpatterns_flag, force_replication_flag = False):
	'''Reinvokes help_merge_internal_hierarchy_of_VTPatternNode_recursive() on returned merged pattern
	whenever merging occurs across immediate children. Re-invocation happens repeatedly until no further merging occurs between immediate children.
	Merging of immediate children of vtn could produce identical sibling subtrees at arbitrary depths within merged subtrees
	due to difference in call-contexts. These produce additional opportunies for further internal merger.

	Reinvocation is done iff retry_merging_on_merged_subpatterns_flag == True.
	'''

	(replication_occured_flag, merging_of_immediate_children_occured, replicated_vtn) = \
			help_merge_internal_hierarchy_of_VTPatternNode_recursive(vtn, retry_merging_on_merged_subpatterns_flag, force_replication_flag)
	assert merging_of_immediate_children_occured == False or replication_occured_flag == True
		# merging_of_immediate_children_occured == True => replication_occured_flag == True

	if retry_merging_on_merged_subpatterns_flag == True and merging_of_immediate_children_occured == True:
		while True:
			print "START: Reinvoking internal subpattern merging!!"
			(RETRIED_replication_occured_flag, RETRIED_merging_of_immediate_children_occured, RETRIED_replicated_vtn) = \
					help_merge_internal_hierarchy_of_VTPatternNode_recursive(replicated_vtn, retry_merging_on_merged_subpatterns_flag, force_replication_flag = False)
			print "END: Reinvoking internal subpattern merging!!"
			assert RETRIED_merging_of_immediate_children_occured == False or RETRIED_replication_occured_flag == True

			if RETRIED_replication_occured_flag == True: #because merging may occur at any lower sublevel (selects only for merging since force_replication_flag == False was used)
				replicated_vtn = RETRIED_replicated_vtn #since additional merging occured somewhere in subtree during latest retry

			if RETRIED_merging_of_immediate_children_occured == False: #no additional merging at immediate children occured during latest retry
				break

	return (replication_occured_flag, merging_of_immediate_children_occured, replicated_vtn)


def merge_internal_hierarchy_of_VTPatternNode(vtn, retry_merging_on_merged_subpatterns_flag = True): #FIXME: need to accept a "merge-pressure" parameter
	'''Return a newly constructed pattern to replace vtn if merging occurs anywhere within vtn tree.
	Return None if no merging occurs anywhere inside vtn.

	retry_merging_on_merged_subpatterns_flag == True => reinvoke merging on any subpattern in which merging occurs, as more merging opportunities may have become available.
	'''
	print "\n ~~~ ATTEMPT INTERNAL MERGE: vtn =", vtn
	(replication_occured_flag, merging_of_immediate_children_occured, replicated_vtn) = \
			retry_help_merge_internal_hierarchy_of_VTPatternNode_recursive(vtn, retry_merging_on_merged_subpatterns_flag, force_replication_flag = False)
	if replication_occured_flag == True:
		return replicated_vtn
	else:
		return None
	
for i, vtr in enumerate(vt_pattern_layer0[:]): #iterate over copy
	internally_merged_vtr = merge_internal_hierarchy_of_VTPatternNode(vtr)
	if internally_merged_vtr != None:
		vt_pattern_layer0[i] = internally_merged_vtr


print "### AFTER INTERNAL MERGES vt_pattern_layer0 = "
print common.list_repr_indented(vt_pattern_layer0)

(merge_layers, final_merge_layer_merged_indices) = merge_layers_of_VTPatternNode_until_stabilized(vt_pattern_layer0, "vt_pattern_layer")


#########################
# VIM sorting

def get_pattern_vim(vtp): #recursive
	'''Compute Variance Impact Metric for tree pattern. This is sum of vims of the first set of tasks encountered along any path from root.
	For example. if the root node is a task, then its vim becomes the pattern vim. Else, sum the recursive pattern-vim for each children path.
	The vim of a path with no tasks is 0.0.
	'''
	if vtp == None:
		return 0.0

	if vtp.get_node_type() == "task":
		pattern_vim = vtp.stats.get_vim()
	else: #examine emanating paths
		pattern_vim = 0.0
		for c in vtp.children:
			if c == None:
				continue
			pattern_vim += get_pattern_vim(c) #recursive call

	return pattern_vim

def cmp_on_pvim(vtp1, vtp2): #descending cmp
	return cmp(get_pattern_vim(vtp2), get_pattern_vim(vtp1))

def sort_children_on_vims_recursive(vtp):
	'''Modify vtp so that the children are rearranged in descending pattern-vims computed for the children. '''

	if vtp == None:
		return

	#recursive, post-order semantics
	for c in vtp.children:
		sort_children_on_vims_recursive(c)
	
	#post order processing of node
	sorted_indexed_children = [(i, c) for i, c in enumerate(vtp.children)]
	sorted_indexed_children.sort( lambda i1_c1, i2_c2: cmp_on_pvim(i1_c1[1], i2_c2[1]) )

	rearranged_children_call_context_sets = [vtp.children_call_context_sets[i] for i, c in sorted_indexed_children]

	vtp.children[:] = [c for i, c in sorted_indexed_children]
	vtp.children_call_context_sets[:] = rearranged_children_call_context_sets


vim_sorted_final_layer = merge_layers[-1][:] #make copy

for vtp in vim_sorted_final_layer: #sort subtrees within patterns (modifies vtp)
	sort_children_on_vims_recursive(vtp)

vim_sorted_final_layer.sort( cmp_on_pvim ) #sort patterns descending


VTPatternNode.enable_stats_printing = True
NodeStatistics.enable_reduced_distribution_printing = True
NodeStatistics.enable_vim_printing = True
print "### VIM Sorted Merged layer (#%s) = " % (len(merge_layers)-1, ), common.list_repr_indented( vim_sorted_final_layer )



#########################
# VIM prioritized trimming

# Idea: Eliminate lowest VIM patterns / pattern-branches in sorted order. Cutoff based on a percentage of cumulative VIM-total till that pattern at that level

vim_cutoff_fraction = 0.10
	# eliminate patterns/pattern-subtrees whose vim is less than vim_cutoff_fraction of the cumulative vim of the preceding patterns at that level
if setting_vim_cutoff_fraction != None:
	vim_cutoff_fraction = setting_vim_cutoff_fraction
print "~~~ USING vim_cutoff_fraction = %s for TRIMMING ~~~" % (vim_cutoff_fraction, )
	


def determine_trim_candidates(list_of_subpatterns, cutoff_fraction):
	'''Given a list of patterns or sub-trees within a node, determines which sub-patterns are significant in terms of pattern-vim or pattern-weight.
	Result is a list of tuples with elements of form (index, pvim, pweight, is_pvim_significant, is_pweight_significant),
	   where index gives location of corresponding subpattern in list_of_subpatterns.
	The resulting list is sorted on the pvim field.
	'''

	result_tuples = []
	for i, subpat in enumerate(list_of_subpatterns):
		pvim = get_pattern_vim(subpat)
		if subpat == None:
			pweight = 0.0
		else:
			pweight = subpat.stats.get_weight()

		result_tuples.append( (i, pvim, pweight, True, True) )

	# Determine trim candidates by pweight
	result_tuples.sort( lambda x, y: cmp(y[2], x[2]) ) #sort descending on pweight
	cumulative_pweight = 0.0
	cutoff_index_by_pweight = len(result_tuples) #initially assume nothing gets cutoff
	for i, (index, pvim, pweight, is_pvim_significant, is_pweight_significant) in enumerate(result_tuples):
		if pweight < cumulative_pweight * cutoff_fraction:
			#cutoff point found
			cutoff_index_by_pweight = i
			break
		cumulative_pweight += pweight

	for i in range(cutoff_index_by_pweight, len(result_tuples)):
		(index, pvim, pweight, is_pvim_significant, is_pweight_significant) = result_tuples[i]
		is_pweight_significant = False
		result_tuples[i] = (index, pvim, pweight, is_pvim_significant, is_pweight_significant)

	# Determine trim candidates by pvim
	result_tuples.sort( lambda x, y: cmp(y[1], x[1]) ) #sort descending on pvim
	cumulative_pvim = 0.0
	cutoff_index_by_pvim = len(result_tuples) #initially assume nothing gets cutoff
	for i, (index, pvim, pweight, is_pvim_significant, is_pweight_significant) in enumerate(result_tuples):
		if pvim < cumulative_pvim * cutoff_fraction:
			#cutoff point found
			cutoff_index_by_pvim = i
			break
		cumulative_pvim += pvim

	for i in range(cutoff_index_by_pvim, len(result_tuples)):
		(index, pvim, pweight, is_pvim_significant, is_pweight_significant) = result_tuples[i]
		is_pvim_significant = False
		result_tuples[i] = (index, pvim, pweight, is_pvim_significant, is_pweight_significant)

	return result_tuples


def trim_pattern_branches_on_significance_recursive(vtp):
	'''
	'''

	#Pre-order elimination
	# Issues:
	#   - Dealing with linear segments: if a child subtree is eliminated, check the following:
	#       + If the deleted child node has containing_linear_segment != None, delete its occurence in all affected ancestors
	
	result_tuples = determine_trim_candidates(vtp.children, vim_cutoff_fraction)
	
	# Now, eliminate children from index cutoff_index onwards
	
	for (i, pvim, pweight, is_pvim_significant, is_pweight_significant) in result_tuples:
		if is_pvim_significant == True or is_pweight_significant == True:
			continue

		#subtree needs to be eliminated
		elim_c = vtp.children[i]
		if elim_c.containing_linear_segment != None:
			originator_node = elim_c.containing_linear_segment[0]
			#locate index of linear segment in originator_node's originating_linear_segments
			index_of_elim_linseg = -1
			for j, linseg_j in enumerate(originator_node.originating_linear_segments):
				if linseg_j == elim_c.containing_linear_segment:
					index_of_elim_linseg = j
					break
			assert index_of_elim_linseg != -1, "trim_pattern_branches_on_significance_recursive(): could not locate linear segment to eliminate in originator_node's originating_linear_segments"

			#delete from originator_node
			originator_node.originating_linear_segments.pop(index_of_elim_linseg)
			originator_node.stats.contribution_stats.pop(index_of_elim_linseg)

			#delete from nodes contained in linear segment
			for contained_node in elim_c.containing_linear_segment[1:]:
				assert contained_node.containing_linear_segment == elim_c.containing_linear_segment
				contained_node.containing_linear_segment = None

		elim_c.parent = None
		vtp.children[i] = "deleted" #placeholder
		vtp.children_call_context_sets[i] = "deleted" #placeholder

	#Make second pass to actually delete elements
	k = 0
	while k < len(vtp.children):
		if vtp.children[k] == "deleted":
			vtp.children.pop(k)
			vtp.children_call_context_sets.pop(k)
		else:
			k += 1

	#recursive on untrimmed children: FIXME: disabled for now
	for c in vtp.children:
		if c == None:
			continue
		trim_pattern_branches_on_significance_recursive(c)


vim_trimmed_final_layer = vim_sorted_final_layer[:] #make copy
for vtp in vim_trimmed_final_layer:
	trim_pattern_branches_on_significance_recursive(vtp)
	sort_children_on_vims_recursive(vtp) #need to resort since trimming of subtrees may have affected the pvims of remaining nodes, and they may no longer be in sorted order at each level

result_tuples = determine_trim_candidates(vim_trimmed_final_layer, vim_cutoff_fraction)
for (i, pvim, pweight, is_pvim_significant, is_pweight_significant) in result_tuples:
	if is_pvim_significant == True or is_pweight_significant == True:
		continue

	#mark pattern for deletion
	vim_trimmed_final_layer[i] = "deleted"

#Make second pass to actually delete
k = 0
while k < len(vim_trimmed_final_layer):
	if vim_trimmed_final_layer[k] == "deleted":
		vim_trimmed_final_layer.pop(k)
	else:
		k += 1

vim_trimmed_final_layer.sort( cmp_on_pvim ) #again sort patterns descending, since trimming may have changed pattern vims
print "### VIM Trimmed layer (#%s) = " % (len(merge_layers)-1, ), common.list_repr_indented( vim_trimmed_final_layer )

vcg_file = open("vcg.txt",'w')
vcg_file.write( common.list_repr_indented( vim_trimmed_final_layer ) )
