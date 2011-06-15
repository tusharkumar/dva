#!/usr/bin/python

# Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
#
# Author: Tushar Kumar, tushardeveloper@gmail.com


import sys
print "~~ SCRIPT INVOCATION:", sys.argv[0]

################ Phase Timing API ################

import time
start_time = time.time()

phase_start_time = None
def MARK_PHASE_START(phase_descr_string):
	global phase_start_time
	phase_start_time = time.time() - start_time

	print "------------- STARTING", phase_descr_string, "at time", phase_start_time, "-------------"

	return phase_start_time



def MARK_PHASE_COMPLETION(phase_descr_string):
	global phase_start_time

	phase_end_time = time.time() - start_time

	print "------------- COMPLETED", phase_descr_string, "at time", (time.time() - start_time), \
		" phase consumed time ", (phase_end_time - phase_start_time), "-------------"

	phase_start_time = None

	return phase_end_time

def get_curr_time():
	return time.time() - start_time



################ Print Indentation Management API ################

indent_depth = 0
def indent_string():
	return (indent_depth*4) * " "

def list_repr_indented(some_list):
	global indent_depth

	if len(some_list) == 0:
		return "[]"

	#Now, list has at least one element
	result  = "[\n"

	indent_depth += 1
	for i, e in enumerate(some_list):
		result += indent_string() + e.__repr__()
		if i+1 < len(some_list):
			result += ","
		result += "\n"
	indent_depth -= 1

	result += indent_string() + "]"
	return result
