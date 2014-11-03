#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>

// Limitation: In a multi-threaded application only one thread must generate profile events.
// The chosen application thread must register its threadid using PROFILE_select_threadid().
// If a threadid has been selected, the various profile dumping routines will query their
// threadid and dump only if found to match the selected threadid.

static int             is_threadid_selected = 0;
static pid_t           selected_threadid; //defined iff is_threadid_selected == 1
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void PROFILE_select_threadid(pid_t threadid)
{
	pthread_mutex_lock(&mutex);
	is_threadid_selected = 1;
	selected_threadid = threadid;
	pthread_mutex_unlock(&mutex);
}

static FILE * prof_fp = 0;
static int fp_counter = 0;
static long long int line_counter = 0;
static const long long int MAX_line_counter = 50000000;

long long dyn_instr_count = 0;

static int dump_profile_setting = 0;
	//dump profile events whenver dump_profile_setting != 0

static int request_next_identifier_flag = 0; //used for identifying functions called via function-pointers

// Will be invoked only from other calls in a thread-safe manner
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

// Assumption: is_threadid_selected and selected_threadid will not change once profile-dumping has been
//   enabled (by the application invoking PROFILE_dump_setting(1)).
//   Hence, only the selected thread can continue past this if-statement in all the API calls, thereby serializing the
//   invocation of all the profile dump functionality. For this reason we don't need to create critical sections
//   in the profile calls.
#define GATE_THREAD                                                                                                      \
{                                                                                                                        \
	if(is_threadid_selected == 1) {                                                                                      \
		pid_t my_threadid = syscall(SYS_gettid);                                                                         \
		if(my_threadid != selected_threadid)                                                                             \
			return;                                                                                                      \
	}                                                                                                                    \
}

//true_of_false = 0 turns off profile dumping,
//  o.w., turn on dumping of profile events
void PROFILE_dump_setting(int true_of_false) {
	pthread_mutex_lock(&mutex);
	dump_profile_setting = true_of_false;
	pthread_mutex_unlock(&mutex);
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
	GATE_THREAD;

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
	GATE_THREAD;

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
	GATE_THREAD;

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
	GATE_THREAD;

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
{
	GATE_THREAD;

	request_next_identifier_flag = 1;
}

#ifdef __cplusplus
} //extern "C"
#endif
