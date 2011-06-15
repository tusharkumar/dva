#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

static FILE * prof_fp = 0;
static int fp_counter = 0;
static long long int line_counter = 0;
static const long long int MAX_line_counter = 50000000;

long long dyn_instr_count = 0;

static int dump_profile_setting = 0;
	//dump profile events whenver dump_profile_setting != 0

static int request_next_identifier_flag = 0; //used for identifying functions called via function-pointers

void PROFILE_open() {
	prof_fp = fopen("profile.dump", "w");
	if(prof_fp == 0) {
		fprintf(stderr, "PROFILE_open: ERROR: Could not open 'profile.dump' for writing");
		exit(1);
	}
}

#define PROFILE_open_next                                                                                                \
{                                                                                                                        \
	char next_filename[100];                                                                                             \
	printf("\nPROFILE_open_next INVOKED, current fp_counter=%d\n", fp_counter);                                          \
	if( fclose(prof_fp) != 0 ) {                                                                                         \
		fprintf(stderr, "PROFILE_open_next: ERROR: Could not close current file handle, fp_counter = %d", fp_counter);   \
		exit(1);                                                                                                         \
	}                                                                                                                    \
	fp_counter++;                                                                                                        \
	sprintf(next_filename, "profile.dump.%d", fp_counter);                                                               \
	prof_fp = fopen(next_filename, "w");                                                                                 \
	if(prof_fp == 0) {                                                                                                   \
		fprintf(stderr, "PROFILE_open_next: ERROR: Could not open '%s' for writing", next_filename);                     \
		exit(1);                                                                                                         \
	}                                                                                                                    \
}

//true_of_false = 0 turns off profile dumping,
//  o.w., turn on dumping of profile events
void PROFILE_dump_setting(int true_of_false) {
	dump_profile_setting = true_of_false;
}

//PROFILE_function_entry(), PROFILE_function_exit(), PROFILE_exception()
//  
//  called_func_id :
//      uniquely identifies what function is being called. Each function
//      is assigned a globally unique integer id
//      
//  function_lex_count : 
//      lexical order of the current call-site in the scope of the containing
//      function
//  
//  loop_lexical_id :
//      identify what inner-most-loop contains this call,
//      0 if call site is outside any loop. loop_lexical_id's are computed
//      locally in each function scope, in depth-first-order
//
//
//  NOTE: For PROFILE_exception(), the above fields reflect the function-call-site
//    at which an exception was caught

void PROFILE_function_entry(
	int called_func_id,
	int function_lex_count,
	int loop_lexical_id
)
{
	if(prof_fp == 0)
		PROFILE_open();

	if(dump_profile_setting == 0)
		return;

	fprintf(prof_fp, "%d %d %d entry %lld\n",
			called_func_id, function_lex_count, loop_lexical_id, dyn_instr_count);
	line_counter++;
	if(line_counter >= MAX_line_counter) {
		PROFILE_open_next;
		line_counter = 0;
	}
}

void PROFILE_function_exit(
	int called_func_id,
	int function_lex_count,
	int loop_lexical_id
)
{
	if(prof_fp == 0)
		PROFILE_open();

	if(dump_profile_setting == 0)
		return;

	fprintf(prof_fp, "%d %d %d exit %lld\n",
			called_func_id, function_lex_count, loop_lexical_id, dyn_instr_count);
	line_counter++;
	if(line_counter >= MAX_line_counter) {
		PROFILE_open_next;
		line_counter = 0;
	}
}

void PROFILE_exception(
	int catching_func_id,
	int function_lex_count,
	int loop_lexical_id
)
{
	if(prof_fp == 0)
		PROFILE_open();

	if(dump_profile_setting == 0)
		return;

	fprintf(prof_fp, "%d %d %d exception %lld\n",
			catching_func_id, function_lex_count, loop_lexical_id, dyn_instr_count);
	line_counter++;
	if(line_counter >= MAX_line_counter) {
		PROFILE_open_next;
		line_counter = 0;
	}
}

//For handling function-pointers
void PROFILE_identifier(int entered_func_id)
{
	if(prof_fp == 0)
		PROFILE_open();

	if(dump_profile_setting == 0)
		return;

	if(request_next_identifier_flag != 0) {
		fprintf(prof_fp, "%d %d %d identifier %lld\n",
				entered_func_id, 0, 0, dyn_instr_count);
		line_counter++;
		if(line_counter >= MAX_line_counter) {
			PROFILE_open_next;
			line_counter = 0;
		}

		request_next_identifier_flag = 0;
	}
}

void PROFILE_request_next_identifier()
{ request_next_identifier_flag = 1; }

#ifdef __cplusplus
} //extern "C"
#endif
