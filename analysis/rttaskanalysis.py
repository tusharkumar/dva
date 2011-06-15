#!/usr/bin/python

# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com


import common

### Profile Construction

import profile

def read_profile_repr(force_max_profile_steps = False, enable_check_if_means_stable = True): #__main__
	global profile_repr

	profile_repr = profile.ProfileRepresentation()
	profile_repr.read("func_map.dump", "func_loop_hier.dump", "profile.dump", specified_force_max_profile_steps = force_max_profile_steps, \
			specified_enable_check_if_means_stable = enable_check_if_means_stable)

### Analysis

###################
# Definitions:
#  1) Call-context for a function-call: The stack of function-calls under which a given function-call was invoked
#
#  2) Minimal Distinguishing call-context: The shortest segment of a function-call's call-context
#      under which it reliably exhibits a specific kind of behavior. Henceforth, referred to as minimal-call-context.
#
#  3) Task: A recurrent minimal-call-context that exhibits highly variable execution times
#     across its multiple invocations.
#
#  4) Variability Component (component for short): A task's variance is divided into components, where each
#     component represents a call-tree invoked by the task that contributes a significant portion of that task's
#     variance.
#
#  5) Variability Contributor: A minimal-call-context that significantly constitutes the variance of a task's component.
#
#  5) CCT: Call-Context Tree
#      - context
#      - function vs. function-node
#
#  6) Immediate contributor of T: A function-node that is directly called by task T, and is a variability contributor of T.
#
#  7) Root contributor of task T: A contributor, possibly contained within other contributors for a task T, which is the
#      underlying cause of the variability contributed by a component I to T.
#
#  8) Task-Variance Graph: A graph representation of tasks determined from an application that captures the
#       call-contexts under which the application exhibits highly variant execution-time behavior, and what
#       call-context contributors contribute to each task's variant behavior.
#
#  9) Types of Variance and Co-variance computation on nodes of CCT:
#     a) Local variance, Var(X): variance computed at a node X w.r.t. that node's invocations. This measures the
#         variability of that node's execution time each time it is invoked.
#
#     b) Local co-variance, C(X): co-variance matrix computed at a node X w.r.t. X's invocations count. This measures the
#         variability and correlation of cumulative execution time spent in each child-node Yi per invocation of X
#            (each Yi may get invoked zero times or multiple times per invocation of X).
#         C(X)_i_j represents the [i, j] term of C(X).
#         Column of co-variance terms C(X)_i is child Yi's co-variance contribution to Var(X).
#
#     c) Contextual Variance, Var_R(X): R is an ancestor of X in the CCT. The variance of X is calculated w.r.t.
#         R's invocations. This measures the variability in the cumulative execution time spent in X per
#         invocation of R.
#
#     d) Contextual Co-variance: C_R(X): R is an ancestor of X in the CCT. The co-variance matrix computed at a node X
#           w.r.t. R's invocation count. This measures the variability and correlation of cumulative execution time
#           spent in each child-node Yi of X per invocation of R.
#         C_R(X)_i_j represents the [i, j] term of C_R(X).
#
#     e) Global Co-variance: G_R(Y1, Y2, ... Yn): R is an ancestor of nodes Y1 .. Yn in CCT. Y1 .. Yn need not be
#        siblings within the same parent.
#
##########
# Identify tasks and dependences, improve task-structure iteratively
#
# Detection Procedure:
#  1) Pre-order traverse CCT to find tasks: Find a high-var CCT node
#     a) if node marked "no-task", skip node: -- cannot be made a task as it lies on a root-contributor-path of some higher-level task
#          - continue traversal
#
#     b) if node is not currently marked as task => mark as task T:
#         Determine composition of T's variability in terms of variability contributed by immediate child-nodes,
#         and the constructive/destructive re-inforcement of the children's variability due to correlation effects.
#           - determine child-nodes that individually contribute significant variability to T, and whose contribution
#               is not dampened by negative correlation effects. => component 'i' and corresponding immediate contributor 'Yi'
#                 + child-node Yi's variability under T's context (C(T)_i_i) is a large fraction of Var(T)
#                              => 'component_fraction'
#                 + sum of child-node's co-variance contribution to T (Sum C(T)_i) is a large fraction of Var(T) -- no damping
#
#     c) Determine whether the variability exhibited by an immediate 'contributor' function-node Yi of T is actually a
#         manifestation of the variability of one or more function-calls invoked from within Yi's call-tree.
#           - pre-order examine Yi's call-tree, as follows:
#
#             function get_lower_root_contributor_candidates(X): X is currently a contributor
#              1. if X is locally low-variant -- can only contribute as a whole
#                       return [] -- failure
#              2. Determine the highest-variance children-nodes of X under X's context
#                    => [Z1 .. Zn] some subset of X's children-nodes (Zi with highest C(X)_i_i)
#              3. chosen_child_var_sum = Sum-over-i-in-[Z1 .. Zn] (C(X)_i_i)
#              4. chosen_child_correlation_sum = Sum-over-i-j-(i!=j)-in-[Z1 .. Zn] (C(X)_i_j)
#              5. If:
#                   + chosen_child_var_sum is a very large fraction (say, >95%) of Var(X)
#                     and
#                   + chosen_child_correlation_sum is small in magnitude compared to chosen_child_var_sum
#                => nodes [Z1 .. Zn] directly capture most of Var(X), and node X does not contribute/dampen
#                      their effect
#                 then: return [Z1 .. Zn] -- success
#                 else: return [] -- failure
#                     
#             procedure search_for_root_contributor(RCF, Y, T): RCF is current root-contributor-frontier for an immediate component Y
#              1. for X in RCF: -- breadth-first-search processing
#                   a. if X is already marked root-contributor for task T:
#                         + continue
#                   b. if X has C_T([Z1 .. Zn]) computed (in X.resume):
#                         + compute chosen_child_var_sum and chosen_child_correlation_sum over [Z1 .. Zn] in T's context
#                         + if chosen_child_var_sum is very large fraction of Var_T(X),
#                                and chosen_child_correlation_sum is comparatively small:
#                                     (!!Optimization: eliminate 2nd test and calculate only self-var terms in T's context)
#                             ++ success!
#                             ++ delete "taskiness" of X if X is currently a task
#                             ++ mark X as (T, no-task)
#                             ++ append [Z1 .. Zn] to RCF (processed by line 1 within same pass)
#                             ++ setup [Z1 .. Zn] as components
#                             ++ delete X from RCF
#                             ++ delete "contributoriness" of X
#                         + else:
#                             ++ mark X as (T, root-contributor, Y)
#                         + continue
#
#                   c. [Z1 .. Zn] = get_lower_root_contributor_candidates(X)
#                   d. if [Z1 .. Zn] == empty-list
#                        + mark X as (T, root-contributor, Y)
#                        + continue
#                   e. setup computation of C_T([Z1 .. Zn]) for next profile pass
#               2. return RCF
#
#            Invocation: search_for_root_contributor(RCF, T) called in each pass for each component I of T with RCF corresponding to I.
#              For first pass, RCF = [Y_I]
# 
# 2) Minimal-Context determination
#     For each function F,
#        match-set = {all Task and Contributor instances of F},
#        other-set = {all significant non-Task and non-Contributor instances of F}
#
# 3) Construct Variance Task Graph (VTG)
#    a) Initial construction: combine Task elements and Contributor elements into single graph-nodes
#         PROVIDED they have similar means & covs:
#          i) if E1 and E2 are Tasks and NOT Contributors, and have same component_indices => same gNode.
#          ii) if E1 and E2 are Tasks AND Contributors, and have same component indices => same gNode.
#          iii) if E1 and E2 are only Contributors => same gNode.
#
#    b) Add Hierarchy edges between gNodes:
#         - if gNode G1 has an element whose parent is an element in gNode G2, then add HierEdge G2 -> G1
#            provided there is no VarEdge G2(task) -> G1 (contrib).
#
#    c) Perform Structural modifications to remove undesirables: cycles, reconvergent-fanouts etc (OPTIONAL)
#
#    d) Apply PatternSimilarityTree to differentiate element call-contexts based on VTG structure:
#        Similarity measure:
#         - Tasks in same gNode are similar, else dissimilar
#         - A pure Contributor and a Task are dissimilar
#         - Contributors in same gNode are similar 
#            Unless (OPTIONAL)
#             + if C1 and C2 contribute widely different fraction's of variances to same (task, component)
#                => split Contributor-gNode
#
#         - Contributors C1 and C2 from different gNodes are dissimilar if they both
#             contribute to the same component of the same Task-gNode.
#             (Criteria: need to distinguish contributors only within the scope of a task)
#
# 4) Augment VTG with average-weight-contributors:
#    a) Examine each CCT task node to determine if individual descendents cumulatively contribute significant
#         (say, >30%) of the *average* execution-time of the task => tag as "weight-contributor" to which (multiple) tasks
#         - hierarchically examine children nodes to find the underlying causes for task's average execution time.
#
#    b) Minimal context determination for identified weight-contributors:
#         For each function F,
#           match-set = {instances of F tagged as "weight-contributors"}
#           other-set = {all significant instances of F not tagged as "weight-contributors"}
#            (NOTE: a node being a task or contributor does not affect its inclusion in other-set)
#
#    c) Add Weight gNodes and Weight Edges to VTG:
#         - Combine weight-contributor CCT nodes into same Weight-gNode if they have similar mean, cov
#         - From each Weight-gNode, add Weight-edges to Task-gNodes to which weight is contributed
#             + edge-annotations-list: (% weight-contributed, frequency with which weight-contributor affects upper task)
#
#    d) Apply PatternSimilarityTree to differentiate weight-contributor element call-contexts based on VTG structure:
#        Similarity measure:
#         - Weight-elements W1 and W2 are similar if they are in same Weight-gNode
#         - W1 and W2 in different Weight-gNodes are dissimilar if they both contribute to the same task, else similar.
#
# 5) Output graph result (dotty)
#       Task nodes:
#          - minimal-call-context
#          - immediate component contributions
#       Contributor nodes:
#          - minimal-call-context
#
#       Edge: between contributor/task to upper-level task
#          - (variance contribution in upper-task's context, frequency with which contributor affects upper task)

### Variance Tagging

import math
def get_node_stats(func_node):
	mean = float(func_node.total_count) / float(func_node.invoke_count)
	var = func_node.sq_err_sum / float(func_node.invoke_count)
	std = math.sqrt(var)
	if mean == 0.0:
		cov = 0.0 # since std must also be 0.0, as execution-times are positive
	else:
		cov = std / mean

	return (mean, var, std, cov)

def tag_var_recursive(func_node):
	if func_node.total_count < profile_repr.min_significant_exec_time:
		return

	(mean, var, std, cov) = get_node_stats(func_node)

	if func_node.invoke_count >= varanalysis.min_invoke_count_needed_for_pattern:
		if cov > varanalysis.cov_high_threshold:
			func_node.analysis["variant"] = "high"

		if cov <= varanalysis.cov_low_threshold:
			func_node.analysis["variant"] = "low"

	for c in func_node.funcs_called:
		if c != None:
			tag_var_recursive(c)


def is_high_var(func_node):
	if func_node.analysis.has_key("variant") and func_node.analysis["variant"] == "high":
		return True
	else:
		return False

def is_low_var(func_node):
	if func_node.analysis.has_key("variant") and func_node.analysis["variant"] == "low":
		return True
	else:
		return False


### Task and Contributor Tagging

import varanalysis

child_var_fraction = 0.3 # Note: for 0.4 got atmost one component due to covar-sum (mpeg2dec)
	# dictates the minimum fraction of variance a child must contribute to its parent
	#  in order for the child for be considered a 'significant' contributor to the parent

most_var_fraction = 0.95
	# dictates the minimum fraction of collective variance that a given
	#  subset of children must be contributing to their parent before they can be considered
	#  to constitute 'most' of the parent's variance

correlation_to_var_insignificance_ratio = (1 - most_var_fraction)
	# 

class Task:
	task_id_to_node = [] # indices task_id -> task_node for all created tasks

	def __init__(self, task_node):
		self.task_id = len(Task.task_id_to_node) #unique id assigned to each created task
		Task.task_id_to_node.append(task_node)

		self.component_indices = []
			# indices of child-nodes in 'func_node.funcs_called' that contribute significant variance
			#  to task 'func_node', and hence form the variability components of the task

		self.component_root_contributor_frontiers = []
			# each element is a list of func_nodes that are the currently known root-contributors for
			#  the component identified by the corresponding element of 'component_indices'

		self.no_task_node_list = []
			# the func_nodes marked as "no_task" due to the current task's search for root-contributors

	def __repr__(self):
		crcf = []
		for rcf in self.component_root_contributor_frontiers:
			func_name_list = [func_node.func_name for func_node in rcf]
			crcf.append(func_name_list)

		no_task_name_list = [func_node.func_name for func_node in self.no_task_node_list]

		result = "Task(%s, ci = %s, crcf = %s, no_task = %s) " \
				% (self.task_id, self.component_indices, crcf, no_task_name_list)
		return result


class Contributor:
	def __init__(self, task_id, index_comp_index, contrib_fraction = None, mean_under_task = None, isRootContributor = False):
		self.task_id           = task_id
		self.index_comp_index  = index_comp_index
		self.contrib_fraction  = contrib_fraction
			#The fraction of consuming task's variance that is contributed by this contributor
		self.mean_under_task   = mean_under_task
			#Mean of the contributor node computed based on the task's invocation count
		self.isRootContributor = isRootContributor

		self.task_covar_indices = None
			# indices into current contributor-node's funcs_called, that have been selected as candidates
			#  for co-variance computation under the given task's context
			#  if None => indices have not yet been selected

		self.task_cross_sq_err_sum = None
			# co-variance matrix result corresponding to entries in 'task_covar_indices'
			#  if None => computation not yet done,
			#  else: is a 2D matrix with sides = len(task_covar_indices)
		
	def __repr__(self):
		result = "Contributor(%s, ici = %s, contrib_frac = %s, mean_under_task = %s, rc = %s, covar_indices = %s, isComputed = %s) " \
				% (self.task_id, self.index_comp_index, self.contrib_fraction, self.mean_under_task, self.isRootContributor, \
					self.task_covar_indices, (self.task_cross_sq_err_sum != None))
		return result


def get_component_indices_of_task(task_node):
	child_var_term_threshold = task_node.sq_err_sum * child_var_fraction
	candidate_child_var_terms = varanalysis.get_sorted_child_var_terms(task_node, child_var_term_threshold)

	candidate_covar_contributions = []
	for (var, i, i2) in candidate_child_var_terms:
		ci_covar_sum = 0.0
		for j in range(len(task_node.cross_sq_err_sum)):
			ci_covar_sum += task_node.cross_sq_err_sum[i][j]
		candidate_covar_contributions.append(ci_covar_sum)

	high_covar_candidate_var_terms = [term for term, ci_covar_sum in zip(candidate_child_var_terms, candidate_covar_contributions) \
										if ci_covar_sum >= child_var_term_threshold]

	# We don't check if immediate contributor's total_count exceeds min_significant_exec_time
	#   => it is possible for a task to have a component without a significant contributor
	
	
	if len(high_covar_candidate_var_terms) > 0: #further prune relatively less significant terms
		order_mag_cutoff = high_covar_candidate_var_terms[0][0] / 10.0
		pruned_candidate_child_var_terms = [term for term in high_covar_candidate_var_terms if term[0] >= order_mag_cutoff]
		#furthermore, if the "internal" term dominates, it will eliminate the child terms
	else:
		pruned_candidate_child_var_terms = high_covar_candidate_var_terms

	final_index_set = [i for (var, i, i2) in pruned_candidate_child_var_terms if i != len(task_node.funcs_called)]
		#cannot have duplicates, since extracted from only diagonal terms.
		# also eliminate internal term, as we don't want to include its effect on total

	return final_index_set

import sys
def associate_contributor(comp_node, task_node, index_comp_index, contrib_fraction, mean_under_task):
	if comp_node.analysis.has_key("contributor"):
		print profile_repr.profile_cet #HACK
		sys.exit("associate_contributor(): ERROR: comp_node.func_name = " + comp_node.func_name + " is already a contributor")

	task_id = task_node.analysis["task"].task_id
	comp_node.analysis["contributor"] = Contributor(task_id, index_comp_index, contrib_fraction, mean_under_task)
	
	
def setup_task(task_node):
	task = task_node.analysis["task"] #instance of Task

	task.component_indices = get_component_indices_of_task(task_node)

	task.component_root_contributor_frontiers = [ [task_node.funcs_called[ci]] for ci in task.component_indices]

	# contributors are immediate children
	for index_comp_index, rcf in enumerate(task.component_root_contributor_frontiers):
		ci = task.component_indices[index_comp_index]
		associate_contributor(rcf[0], task_node, index_comp_index, task_node.cross_sq_err_sum[ci][ci] / task_node.sq_err_sum, float(rcf[0].total_count) / float(task_node.invoke_count))
	
def remove_task(task_node):
	task = task_node.analysis["task"] #instance of Task

	# delete contributor info
	for rcf in task.component_root_contributor_frontiers:
		for x in rcf:
			if not x.analysis.has_key("contributor"):
				sys.exit("remove_task(): ERROR: node does not have 'contributor' analysis field, but is a contributor for task_id = " \
					+ str(task.task_id))

			del x.analysis["contributor"]

	# clear "no_task" tags
	for x in task.no_task_node_list:
		if not x.analysis.has_key("task"):
			sys.exit("remove_task(): ERROR: node does not have 'task' analysis field, but is present in 'no_task_node_list' for task_id = " \
					+ str(task.task_id))

		if x.analysis["task"] != "no_task":
			sys.exit("remove_task(): ERROR: node does not have 'no_task' tag but is present in 'no_task_node_list' for task_id = " \
					+ str(task.task_id))

		# Now, x.analysis["task"] is defined, and is = "no_task"
		del x.analysis["task"]

	del task_node.analysis["task"]

	

def get_lower_root_contributor_candidates(contrib_node, min_significant_exec_time):
	if is_low_var(contrib_node):
		return []

	child_var_term_threshold = contrib_node.sq_err_sum * child_var_fraction
	highest_local_var_internal_and_children = varanalysis.get_sorted_child_var_terms(contrib_node, child_var_term_threshold)
	highest_local_var_children = [term for term in highest_local_var_internal_and_children \
									if term[1] != len(contrib_node.funcs_called)]
		#eliminate internal term

	chosen_child_var_sum = 0.0
	for (var, i, i2) in highest_local_var_children:
		chosen_child_var_sum += var

	chosen_child_indices = [i for (var, i, i2) in highest_local_var_children]

	#None of the chosen children should be below a min_significant_exec_time threshold
	anyBelow = False
	for i in chosen_child_indices:
		if contrib_node.funcs_called[i].total_count < min_significant_exec_time:
			anyBelow = True
			break
	if anyBelow == True:
		return []


	chosen_child_correlation_sum = 0.0
	for i in chosen_child_indices:
		for j in chosen_child_indices:
			if i != j:
				chosen_child_correlation_sum += contrib_node.cross_sq_err_sum[i][j]

	if (chosen_child_var_sum > 0.0 and chosen_child_var_sum >= most_var_fraction * contrib_node.sq_err_sum) \
			and (abs(chosen_child_correlation_sum) / chosen_child_var_sum < correlation_to_var_insignificance_ratio):
		# chosen children must constitute most of parent's variance, without being affected
		#   by correlation effects in the parent's context

		return chosen_child_indices

	else:
		return []


def search_for_root_contributor(rcf, task_node, min_significant_exec_time):
	global num_task_covar_compute_requests

	task = task_node.analysis["task"] #instance of Task

	ix = -1
	while ix + 1 < len(rcf): #breadth-first-search queue
		ix += 1

		x = rcf[ix]
		x_contrib = x.analysis["contributor"]

		if x_contrib.isRootContributor:
			continue

		##

		if x_contrib.task_covar_indices != None: #computation request put in during previous pass
			if x_contrib.task_cross_sq_err_sum == None:
				sys.exit("search_for_root_contributor(): ERROR: request from previous pass unfulfilled:" \
						+ " cross-co-var computation in task's context" \
						+ " task_id = %s contributor-func_name = %s" % (task_node.analysis["task"].task_id, x.func_name))

			# Now, task-contextual co-variance computation done

			task_chosen_child_var_sum = 0.0
			for i in range(len(x_contrib.task_cross_sq_err_sum)):
				task_chosen_child_var_sum += x_contrib.task_cross_sq_err_sum[i][i]

			task_chosen_child_correlation_sum = 0.0
			for i in range(len(x_contrib.task_cross_sq_err_sum)):
				for j in range(len(x_contrib.task_cross_sq_err_sum)):
					if i != j:
						task_chosen_child_correlation_sum += x_contrib.task_cross_sq_err_sum[i][j]

			if (task_chosen_child_var_sum >= most_var_fraction * task_node.sq_err_sum) \
					and (abs(task_chosen_child_correlation_sum) / task_chosen_child_var_sum < correlation_to_var_insignificance_ratio):
				if x.analysis.has_key("task"):
					if x.analysis["task"] != "no_task":
						remove_task(x)
						x.analysis["task"] = "no_task"
						task.no_task_node_list.append(x)
				else:
					x.analysis["task"] = "no_task"
					task.no_task_node_list.append(x)

				new_lower_level_contrib_nodes = [x.funcs_called[ci] for ci in x_contrib.task_covar_indices]
				rcf.extend(new_lower_level_contrib_nodes) #add x's children to rcf
				for iy, y in enumerate(new_lower_level_contrib_nodes):
					contrib_fraction = x_contrib.task_cross_sq_err_sum[iy][iy] / task_node.sq_err_sum
					mean_under_task = float(y.total_count) / float(task_node.invoke_count)
					associate_contributor(y, task_node, x_contrib.index_comp_index, contrib_fraction, mean_under_task)

				del rcf[i] # remove x from rcf
				del x.analysis["contributor"]

				ix -= 1 #since deleted an element
				continue

			else:
				x_contrib.isRootContributor = True
				continue

		##

		# co-variance computation not done yet, set it up for next profile pass

		lower_level_candidate_indices = get_lower_root_contributor_candidates(x, min_significant_exec_time)

		if len(lower_level_candidate_indices) == 0:
			x_contrib.isRootContributor = True
			continue

		x_contrib.task_covar_indices = lower_level_candidate_indices #setup
		num_task_covar_compute_requests += 1

	return rcf




def process_task_dependences(task_node, min_significant_exec_time):
	task = task_node.analysis["task"] #instance of Task

	for i in range(len(task.component_root_contributor_frontiers)):
		task.component_root_contributor_frontiers[i] = \
				search_for_root_contributor(task.component_root_contributor_frontiers[i], task_node, min_significant_exec_time)
		


def check_and_make_task(func_node, min_significant_exec_time):
	if not func_node.analysis.has_key("task"):
		if is_high_var(func_node):
			func_node.analysis["task"] = Task(func_node)
			setup_task(func_node)
		else:
			return #Failed to make task

	if func_node.analysis["task"] == "no_task":
		return #Forbidden from being made a task

	#Now, func_node was already a task or was just made one

	process_task_dependences(func_node, min_significant_exec_time)


def tag_tasks_top_down_recursive(func_node, min_significant_exec_time):
	if func_node.total_count < min_significant_exec_time:
		return

	check_and_make_task(func_node, 0) #FIXME: min_significant_exec_time)

	for c in func_node.funcs_called:
		if c != None:
			tag_tasks_top_down_recursive(c, min_significant_exec_time)


### Computing Task-context co-variance

def compute_task_context_covariance_on_node_visit(curr_node, count_diff, children_exec_counts):
	if curr_node.analysis.has_key("contributor"):
		contributor = curr_node.analysis["contributor"]

		if contributor.task_covar_indices != None: #have a request
			incr_required_children_exec_counts = [children_exec_counts[ci] for ci in contributor.task_covar_indices]

			if "required_children_exec_counts" not in dir(contributor):
				contributor.required_children_exec_counts = len(incr_required_children_exec_counts) * [0.0]

			for i in range(len(incr_required_children_exec_counts)):
				contributor.required_children_exec_counts[i] += incr_required_children_exec_counts[i]


	if curr_node.analysis.has_key("task") and curr_node.analysis["task"] != "no_task":
		task = curr_node.analysis["task"]

		task_invoke_count = curr_node.invoke_count

		# compute co-variances independently at each contributor
		for rcf in task.component_root_contributor_frontiers:
			for x_contrib_node in rcf:
				x_contributor = x_contrib_node.analysis["contributor"]

				if x_contributor.isRootContributor or x_contributor.task_covar_indices == None:
					# either a root-contributor => any compute request would have been satisfied in some previous pass
					#  or, no request has been made for this pass
					continue

				required_child_nodes = [x_contrib_node.funcs_called[ci] for ci in x_contributor.task_covar_indices]
				required_means = [float(c.total_count) / float(task_invoke_count) for c in required_child_nodes]

				if x_contributor.task_cross_sq_err_sum == None:
					matrix_size = len(x_contributor.task_covar_indices)
					x_contributor.task_cross_sq_err_sum = matrix_size * [None]
					for i in range(len(x_contributor.task_cross_sq_err_sum)):
						x_contributor.task_cross_sq_err_sum[i] = matrix_size * [0.0]

				if "required_children_exec_counts" in dir(x_contributor):
					for i, (ici_exec_count, ici_mean) in enumerate( zip(x_contributor.required_children_exec_counts, required_means) ):
						for j, (jci_exec_count, jci_mean) in enumerate( zip(x_contributor.required_children_exec_counts, required_means) ):
							x_contributor.task_cross_sq_err_sum[i][j] \
									+= (ici_exec_count - ici_mean) * (jci_exec_count - jci_mean)

					del x_contributor.required_children_exec_counts



### Invocation

def identify_task_structure_and_propagate_dependences_on_cct(): #__main__
	global num_task_covar_compute_requests

	common.MARK_PHASE_START("Variance Tagging")
	tag_var_recursive(profile_repr.profile_cet)
	common.MARK_PHASE_COMPLETION("Variance Tagging")


	print "Task Tagging Parameters:"
	print "  child_var_fraction =", child_var_fraction
	print "  most_var_fraction =", most_var_fraction
	print "  correlation_to_var_insignificance_ratio = ", correlation_to_var_insignificance_ratio
	print

	max_tagging_passes = 4

	task_tagging_pass_number = 0
	while task_tagging_pass_number < max_tagging_passes:
		task_tagging_pass_number += 1
		num_task_covar_compute_requests = 0

		common.MARK_PHASE_START("Task Tagging Pass #" + str(task_tagging_pass_number))
		tag_tasks_top_down_recursive(profile_repr.profile_cet, profile_repr.min_significant_exec_time)
			#calculates num_task_covar_compute_requests
		common.MARK_PHASE_COMPLETION("Task Tagging Pass #" + str(task_tagging_pass_number))

		if num_task_covar_compute_requests > 0:
			profile.run_user_profile_pass(profile_repr, compute_task_context_covariance_on_node_visit)
		else:
			break

	print profile_repr.profile_cet


### Dumping task-cet

class ExtractTree:
	'''Captures nodes of interest from CET and their corresponding structure.
	CET nodes of interest are classified as: Task, Contributor, Contrast.
	An entire subtree of nodes-of-interest becomes a separate ExtractTree (hierarchical)
	'''

	id_count = 0

	def __init__(self, id = None, func_name = None, total_count = None, invoke_count = None, mean = None, cov = None, \
					type = None, contributesVariance = False, contrib_fraction = None, mean_under_task = None, contribution_callcontext = None, extracted_children_tuples = None):

		self.func_node = None # Needs to be defined only to extract statistics from func_node

		self.extract_parent = None

		self.id = id
		if self.id == None:
			self.id = ExtractTree.id_count
			ExtractTree.id_count = ExtractTree.id_count + 1

		#Statistics from func-node
		self.func_name = func_name
		self.total_count = total_count
		self.invoke_count = invoke_count
		self.mean = mean
		self.cov = cov

		#Info specific to Task/Contributor/Contrast
		self.type = type #"task"/"contributor"/"contrast"

		self.contributesVariance = contributesVariance # always True for "contributor", maybe True for "task", always False for "contrast"

		#defined iff self.contributesVariance == True
		self.contrib_fraction = contrib_fraction
		self.mean_under_task  = mean_under_task
		self.contribution_callcontext = None
		if contribution_callcontext != None:
			self.contribution_callcontext = contribution_callcontext[:]

		self.extracted_children_tuples = [] #tuples of form (call-context from child to this node, instance of ExtractTree holding child)
			# Assumption 1: There are no duplicate call-contexts
			# Assumption 2: No call-context contains another call-context

		if extracted_children_tuples != None:
			self.extracted_children_tuples = extracted_children_tuples[:]

	def associate(self, func_node, extract_parent, type, contributesVariance = False, contrib_fraction = None, mean_under_task = None, contribution_callcontext = None):
		self.func_node = func_node
		self.extract_parent = extract_parent
		self.instantiate_stats()

		if type != "task" and type != "contributor" and type != "contrast":
			sys.exit("ExtractTree::associate(): type = %s is invalid" % (type, ))
		self.type = type
		if type == "contrast" and contributesVariance == True:
			sys.exit("ExtractTree::associate(): contrast node cannot contribute variance")
		if type == "contributor" and contributesVariance == False:
			sys.exit("ExtractTree::associate(): contributor node must contribute variance")
		self.contributesVariance = contributesVariance
		if self.contributesVariance == True:
			self.contrib_fraction = contrib_fraction
			self.mean_under_task  = mean_under_task
			self.contribution_callcontext = contribution_callcontext
		else:
			self.contrib_fraction = None
			self.mean_under_task  = None
			self.contribution_callcontext = None

		return self

	def instantiate_stats(self):
		self.func_name = self.func_node.func_name
		self.total_count = self.func_node.total_count
		self.invoke_count = self.func_node.invoke_count
		(self.mean, var, std, self.cov) = get_node_stats(self.func_node)

	def add_child(self, descendent_func_node, type, contributesVariance = False, contrib_fraction = None, mean_under_task = None, contribution_callcontext = None):
		"Returns a tuple: (call-context-from-child-to-here, newly allocated ExtractTree node for child)"
		print "add_child() invoked"
		desc_full_cc = descendent_func_node.get_call_chain_context()
		stop_index = None
		##WARNING: Comparing by name is fallible to the same name appearing multiple times in a call-chain (say, due to recursion or NULL_FUNCs)
		#for i, (fn, fli) in enumerate(desc_full_cc):
		#	if fn == self.func_node.func_name: #found way back to current node
		#		stop_index = i # include call-context till before this index
		#		break

		# Instead, more robustly compare on nodes from CCT
		desc_call_chain = [descendent_func_node]
		while desc_call_chain[-1].parent_func_node != None:
			desc_call_chain.append( desc_call_chain[-1].parent_func_node )
		assert len(desc_call_chain) == len(desc_full_cc)

		for i, ancestor_func_node in enumerate(desc_call_chain):
			assert ancestor_func_node.func_name == desc_full_cc[i][0]
			if ancestor_func_node == self.func_node: #found way back to current node
				stop_index = i # include call-context till before this index
				break

		if stop_index == None:
			sys.exit("ExtractTree::add_child(): ERROR: given func_node is not a descendent")

		desc_cc_to_here = desc_full_cc[0:stop_index]

		#error check
		for cc, et in self.extracted_children_tuples:
			if cc == desc_cc_to_here:
				sys.exit("ExtractTree::add_child(): ERROR: child context already exists\n desc_cc_to_here=%s\n desc_full_cc=%s\n self.extracted_children_tuples=%s\n" \
								% (desc_cc_to_here, desc_full_cc, self.extracted_children_tuples) )

		extract_descendent = ExtractTree().associate(descendent_func_node, self, type, contributesVariance, contrib_fraction, mean_under_task, contribution_callcontext)
		self.extracted_children_tuples.append( (desc_cc_to_here, extract_descendent) )
		return self.extracted_children_tuples[-1] # the tuple just added

	def get_node_type(self):
		return self.type

	def get_func_name(self):
		return self.func_name

	def get_HACK_stats(self):
		return (self.mean, self.cov)

	def get_contributesVariance(self):
		return self.contributesVariance

	def get_extract_tree_ids(self):
		return self.id

	def __repr__(self):
		children_repr  = "["
		common.indent_depth += 1
		for i, (cc, et_child) in enumerate(self.extracted_children_tuples):
			children_repr += "\n" + common.indent_string() + (cc, et_child).__repr__()
			if i+1 < len(self.extracted_children_tuples):
				children_repr += ","
		common.indent_depth -= 1

		if len(self.extracted_children_tuples) > 0:
			children_repr += "\n" + common.indent_string()
		children_repr += "]"

		result = 'ExtractTree(%s, "%s", %s, %s, %s, %s, "%s", %s, %s, %s, %s, %s)' \
					% (self.id, self.func_name, self.total_count, self.invoke_count, self.mean, self.cov, \
						self.type, self.contributesVariance, self.contrib_fraction, self.mean_under_task, self.contribution_callcontext, children_repr)

		return result


def identify_task_func_names(): #__main__
	global task_func_names

	task_func_names = []
	for task_node in Task.task_id_to_node:
		if task_node.analysis.has_key("task") and task_node.analysis["task"] != "no_task":
			if task_node.func_name not in task_func_names:
				task_func_names.append(task_node.func_name)


def dump_task_cet_edge_helper(fhandle, curr_node, curr_node_number, last_parent_node_dumped_number, last_parent_node_dumped_type):
	if last_parent_node_dumped_number == None:
		return

	annot_string = "" #"(%.3f, %.3f, %.3f)"
	edge_label = 'label="%s"' % (annot_string)
	fhandle.write( '%s -> %s [%s] [style=bold];\n' % (last_parent_node_dumped_number, curr_node_number, edge_label) )

dump_node_count = 0
def dump_task_cet_node_helper(fhandle, curr_node, last_parent_node_dumped_number, last_parent_node_dumped_type, extracted_trees, curr_extract_tree_parent):
	'''
	Pre-order CET processing algorithm to extract the variant task structure of the application. Produces and dumps subtrees capturing each instance
	of variance-related behavior. The intent is to capture the functions capable of exhibiting significant variant behavior (including contrasting between
	instances that are variant and non-variant), and the structural context under which such behavior is observed.

		fhandle                       :  the graphviz-dot file-handle to dump the extracted trees to as they are being discovered
		curr_node                     :  the current CET node being processed. Must be the root node to start with

		Recursive information for last dumped node: needed to draw an edge from the last node as parent to current node being dumped
			last_parent_node_dumped_number:  a counter to uniquely label each node dumped in the graphviz-dot file
			last_parent_node_dumped_type  :

		extracted_trees               : return value, the list of trees discoved in the CET during the dumping process
		curr_extract_tree_parent      : Recursive information about the ExtractTree object allocated for the last node dumped
	                                      if it was an ancestor node of current node
	'''
	global dump_node_count

	next_parent_node_dumped_number = last_parent_node_dumped_number
	next_parent_node_dumped_type = last_parent_node_dumped_type
	next_extract_tree_parent = curr_extract_tree_parent

	(mean, var, std, cov) = get_node_stats(curr_node)
	if curr_node.analysis.has_key("task") and curr_node.analysis["task"] != "no_task": #task, possibly also contributor
		task = curr_node.analysis["task"]

		#print the task
		node_number = dump_node_count
		dump_node_count = dump_node_count + 1

		label_parm = '{%s: %s' % (task.task_id, curr_node.func_name)
		label_parm += '| mean=%.2f invoke_count=%.2f CoV=%.2f' % (mean, curr_node.invoke_count, cov)
		label_parm += '}'
		display_parms = " shape=Mrecord style=bold"
		decorator = 'label="%s" %s' % (label_parm, display_parms)
		fhandle.write( '%s [%s];\n' % (node_number, decorator) )

		if curr_node.analysis.has_key("contributor"): #both a task and a contributor
			contributesVariance = True
			contributor = curr_node.analysis["contributor"]
			contrib_fraction = contributor.contrib_fraction
			mean_under_task  = contributor.mean_under_task
			contribution_callcontext = curr_node.get_call_chain_context_to_named_ancestor( Task.task_id_to_node[ contributor.task_id ].func_name )
		else:
			contributesVariance = False
			contrib_fraction = None
			mean_under_task  = None
			contribution_callcontext = None

		if curr_extract_tree_parent != None:
			cc, next_extract_tree_parent = curr_extract_tree_parent.add_child(curr_node, "task", contributesVariance, contrib_fraction, mean_under_task, contribution_callcontext)
		else:
			next_extract_tree_parent = ExtractTree().associate(curr_node, None, "task", contributesVariance, contrib_fraction, mean_under_task, contribution_callcontext)
		#FIXME: pass Task specific info

		dump_task_cet_edge_helper(fhandle, curr_node, node_number, last_parent_node_dumped_number, last_parent_node_dumped_type)

		next_parent_node_dumped_number = node_number
		next_parent_node_dumped_type = "task"

	elif curr_node.analysis.has_key("contributor"): #purely a contributor
		contributor = curr_node.analysis["contributor"]

		#print the contributor
		node_number = dump_node_count
		dump_node_count = dump_node_count + 1

		label_parm = '{%s' % (curr_node.func_name, )
		label_parm += '| mean=%.2f invoke_count=%.2f CoV=%.2f' % (mean, curr_node.invoke_count, cov)
		label_parm += '}'
		display_parms = " shape=Mrecord style=dotted"
		decorator = 'label="%s" %s' % (label_parm, display_parms)
		fhandle.write( '%s [%s];\n' % (node_number, decorator) )

		contribution_callcontext = curr_node.get_call_chain_context_to_named_ancestor( Task.task_id_to_node[ contributor.task_id ].func_name )
		if curr_extract_tree_parent != None:
			cc, next_extract_tree_parent = curr_extract_tree_parent.add_child(curr_node, "contributor", True, contributor.contrib_fraction, contributor.mean_under_task, contribution_callcontext)
		else:
			next_extract_tree_parent = ExtractTree().associate(curr_node, None, "contributor", True, contributor.contrib_fraction, contributor.mean_under_task, contribution_callcontext)

		dump_task_cet_edge_helper(fhandle, curr_node, node_number, last_parent_node_dumped_number, last_parent_node_dumped_type)

		next_parent_node_dumped_number = node_number
		next_parent_node_dumped_type = "contributor"

	elif curr_node.func_name in task_func_names: #there are task nodes with this name, dump as contrast
		#print
		node_number = dump_node_count
		dump_node_count = dump_node_count + 1

		label_parm = '{%s' % (curr_node.func_name, )
		label_parm += '| mean=%.2f invoke_count=%.2f CoV=%.2f' % (mean, curr_node.invoke_count, cov)
		label_parm += '}'
		display_parms = " shape=Mrecord style=diagonals"
		decorator = 'label="%s" %s' % (label_parm, display_parms)
		fhandle.write( '%s [%s];\n' % (node_number, decorator) )

		if curr_extract_tree_parent != None:
			cc, next_extract_tree_parent = curr_extract_tree_parent.add_child(curr_node, "contrast")
		else:
			next_extract_tree_parent = ExtractTree().associate(curr_node, None, "contrast")

		dump_task_cet_edge_helper(fhandle, curr_node, node_number, last_parent_node_dumped_number, last_parent_node_dumped_type)

		next_parent_node_dumped_number = node_number
		next_parent_node_dumped_type = "contrast"

	if curr_extract_tree_parent == None and next_extract_tree_parent != None: # a new ExtractTree is rooted here
		extracted_trees.append(next_extract_tree_parent)

	for c in curr_node.funcs_called:
		if c != None:
			dump_task_cet_node_helper(fhandle, c, next_parent_node_dumped_number, next_parent_node_dumped_type, extracted_trees, next_extract_tree_parent)

def dump_task_cet(cet_root, taskcet_filename = "taskcet.dot"):
	"Dumps extracted CET task-set, and returns a list of ExtractTree representing the same"

	extracted_trees = []
	#print header

	fhandle = open(taskcet_filename, "w")
	fhandle.write('digraph G {\n')
	#fhandle.write('  node [shape=record style=filled];\n')
	#fhandle.write('  edge [color=grey75];\n')
	fhandle.write('  ranksep="1.0 equally";\n')

	global dump_node_count
	dump_node_count = 0
	dump_task_cet_node_helper(fhandle, cet_root, None, None, extracted_trees, None)

	#print footer
	fhandle.write("}\n")
	fhandle.close()

	return extracted_trees
	
def construct_and_dump_extracted_trees(): #__main__
	global extracted_trees

	extracted_trees = dump_task_cet(profile_repr.profile_cet)
	print "extracted_trees = ", extracted_trees

	et_filename = "extracted_trees.txt"
	et_fhandle = open(et_filename, "w")
	et_fhandle.write( common.list_repr_indented(extracted_trees) )
	et_fhandle.close()

	maincc_filename = "maincc.txt"
	maincc_fhandle = open(maincc_filename, "w")
	for et in extracted_trees:
		maincc_fhandle.write("maincc[%s] = %s\n" % (et.id, et.func_node.get_call_chain_context()))
	maincc_fhandle.close()



#######

def run_analysis(force_max_profile_steps = False, enable_check_if_means_stable = True):
	read_profile_repr(force_max_profile_steps, enable_check_if_means_stable)
	identify_task_structure_and_propagate_dependences_on_cct()
	identify_task_func_names()
	construct_and_dump_extracted_trees()

if __name__ == "__main__":
	run_analysis()
else:
	print "~~ IMPORTED rttaskanalysis ~~"
