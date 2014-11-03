// Copyright 2011, Tushar Kumar, Georgia Institute of Technology, under the 3-clause BSD license
//
// Author: Tushar Kumar, tushardeveloper@gmail.com


#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace llvm;


namespace {
	int funcs_named_count = 0;
	std::map<std::string, int> map_func_name_to_id;

	GlobalVariable * dyn_instr_count_global = 0;
	Function * PROFILE_function_entry = 0;
	Function * PROFILE_function_exit = 0;
	Function * PROFILE_exception = 0;
	Function * PROFILE_dump_setting = 0;
	Function * PROFILE_identifier = 0;
	Function * PROFILE_request_next_identifier = 0;

	std::ofstream ofs_func_map;
	std::ofstream ofs_func_loop_hier;


	class DVAModuleSetup : public ModulePass {
	public:
		static char ID;
		DVAModuleSetup() : ModulePass(ID) {}

		virtual bool runOnModule(Module &M) {
			std::cerr << "--------------DVAModuleSetup--------------" << std::endl;
			dyn_instr_count_global = new GlobalVariable(M, Type::getInt64Ty(M.getContext()), false, GlobalValue::ExternalLinkage, 0, "dyn_instr_count");
			
			//Insert declarations for PROFILE functions
			std::vector<const Type *> type_3int_params;
			type_3int_params.push_back(Type::getInt32Ty(M.getContext()));
			type_3int_params.push_back(Type::getInt32Ty(M.getContext()));
			type_3int_params.push_back(Type::getInt32Ty(M.getContext()));

			FunctionType * func_type_3ints = FunctionType::get(Type::getVoidTy(M.getContext()), type_3int_params, false);

			PROFILE_function_entry = Function::Create(func_type_3ints, GlobalValue::ExternalLinkage, "PROFILE_function_entry", &M);
			PROFILE_function_exit = Function::Create(func_type_3ints, GlobalValue::ExternalLinkage, "PROFILE_function_exit", &M);
			PROFILE_exception = Function::Create(func_type_3ints, GlobalValue::ExternalLinkage, "PROFILE_exception", &M);


			std::vector<const Type *> type_params_1int;
			type_params_1int.push_back(Type::getInt32Ty(M.getContext()));

			FunctionType * func_type_1int = FunctionType::get(Type::getVoidTy(M.getContext()), type_params_1int, false);

			PROFILE_dump_setting = Function::Create(func_type_1int, GlobalValue::ExternalLinkage, "PROFILE_dump_setting", &M);
			PROFILE_identifier = Function::Create(func_type_1int, GlobalValue::ExternalLinkage, "PROFILE_identifier", &M);

			std::vector<const Type *> type_params_0int;

			FunctionType * func_type_0int = FunctionType::get(Type::getVoidTy(M.getContext()), type_params_0int, false);

			PROFILE_request_next_identifier = Function::Create(func_type_0int, GlobalValue::ExternalLinkage, "PROFILE_request_next_identifier", &M);

			return true;
		}
	};

	//Global Definitions
	char DVAModuleSetup::ID = 0;
	RegisterPass<DVAModuleSetup> XM("dva_profileinstrument_setup", "Inserts Profiling Declarations into a module for Dominant Variance Analysis");

	class DVAProf : public FunctionPass {
	public:
		static char ID;
		DVAProf() : FunctionPass(ID) {}

		virtual void getAnalysisUsage(AnalysisUsage &AU) const {
			AU.addRequired<DVAModuleSetup>();
			AU.addRequired<LoopInfo>();
		}

		virtual bool runOnFunction(Function &F) {
			std::cerr << "--------------DVAProf--------------" << std::endl;
			std::cerr << "F = "; errs() << F; std::cerr << std::endl;
			if(F.isDeclaration()) {
				std::cerr << "**EXTERNAL**" << std::endl;;
				return false;
			}

			if( isSpecial_PROFILE_function(F.getName()) )
				return false;

			Module& M = *(F.getParent());
			static bool isFirstInvoke = true;
			if(isFirstInvoke) {
				isFirstInvoke = false;
				setup_parms(M);
			}

			std::string curr_func_name = F.getName();
			if(curr_func_name == "")
				curr_func_name = "NO_NAME";
			add_func_name_to_map(curr_func_name);
				
			int curr_func_id = map_func_name_to_id[curr_func_name];



			LoopInfo &LI = getAnalysis<LoopInfo>();
			LI.print(errs()); //std::cerr);

			std::vector<Loop *> vLoops;
			std::map<Loop *, int> map_loop_to_lexical_id;
			
			//First element is entire function, outside any loops, represented by null
			vLoops.push_back((Loop *) 0);
			map_loop_to_lexical_id[(Loop *) 0] = 0;

			ofs_func_loop_hier << std::string(F.getName()) << " = ";
			construct_and_dump_func_loop_hier(LI, vLoops, map_loop_to_lexical_id);

			for(int i=0; i<(int)vLoops.size(); i++) {
				std::cerr << "vLoops[" << i << "] = " << vLoops[i] << std::endl;
			}

			int function_lex_count = 0;
			//for each basic-block in function
			for (Function::iterator b = F.begin(), be = F.end(); b != be; b++) {
				std::cerr << "Size of bb is " << b->size() << std::endl;

				Loop * innermost_containing_loop = LI.getLoopFor(&*b);
				assert(map_loop_to_lexical_id.count(innermost_containing_loop) == 1);
				int loop_lexical_id = map_loop_to_lexical_id[innermost_containing_loop];

				int instr_count = 0;
				//for each instruction in basic-block
				for(BasicBlock::iterator i=b->begin(), ie=b->end(); i != ie; i++) {
					if(InvokeInst * invokeInst = dyn_cast<InvokeInst>(i)) {
						//FIXME: "NULL" functions represent calls-to-function-pointers. Need special handling!
						std::string invoked_func_name = "NULL_FUNC";
						if(invokeInst->getCalledFunction() != 0)
							invoked_func_name = invokeInst->getCalledFunction()->getName();
						if(invoked_func_name == "")
							invoked_func_name = "NO_NAME";

						//test needed because instrumenting invokeInst creates basic blocks containing special profile calls; also programmer may insert some calls
						if( isSpecial_PROFILE_function(invoked_func_name) )
							continue;

						add_func_name_to_map(invoked_func_name);

						//add instr_count to dyn_instr_count_global, set instr_count = 0
						update_dyn_instr_count(instr_count, i);
						instr_count = 0;

						int invoked_func_id = map_func_name_to_id[invoked_func_name];

						//insert profile-function-entry call before call-site
						insert_profile_function_entry_call(invoked_func_id, function_lex_count, loop_lexical_id, i);
						if(invoked_func_name == "NULL_FUNC")
							insert_profile_request_next_identifier(i); //need function called via function-pointer to identify itself

						//insert basic-block containing profile-function-exit call on normal path
						BasicBlock * orig_normalDestBB = invokeInst->getNormalDest(); //dest when invoked-function returns normally
						BasicBlock * new_normalDestBB = BasicBlock::Create(F.getContext(), "", &F, orig_normalDestBB);
						BranchInst * new_normal_br = BranchInst::Create(orig_normalDestBB, new_normalDestBB); //make 'new' branch to 'orig'
						insert_profile_function_exit_call(invoked_func_id, function_lex_count, loop_lexical_id, new_normal_br);
						invokeInst->setNormalDest(new_normalDestBB);
						
						//insert basic-block containing profile-exception call on unwind path
						BasicBlock * orig_unwindDestBB = invokeInst->getUnwindDest();
						BasicBlock * new_unwindDestBB = BasicBlock::Create(F.getContext(), "", &F, orig_unwindDestBB);
						BranchInst * new_unwind_br = BranchInst::Create(orig_unwindDestBB, new_unwindDestBB); //make 'new' branch to 'orig'
						insert_profile_exception_call(invoked_func_id, function_lex_count, loop_lexical_id, new_unwind_br);
						invokeInst->setUnwindDest(new_unwindDestBB);

						function_lex_count++;
					}
					
					else if(CallInst * callInst = dyn_cast<CallInst>(i)) {
						//FIXME: "NULL" functions represent calls-to-function-pointers. Need special handling!
						std::string called_func_name = "NULL_FUNC";
						if(callInst->getCalledFunction() != 0)
							called_func_name = callInst->getCalledFunction()->getName();
						if(called_func_name == "")
							called_func_name = "NO_NAME";

						//test needed because instrumenting invokeInst creates basic blocks containing special profile calls; also programmer may insert some calls
						if( isSpecial_PROFILE_function(called_func_name) )
							continue;

						add_func_name_to_map(called_func_name);

						//add instr_count to dyn_instr_count_global, set instr_count = 0
						update_dyn_instr_count(instr_count, i);
						instr_count = 0;

						int called_func_id = map_func_name_to_id[called_func_name];

						//insert profile-function-entry call before call-site
						insert_profile_function_entry_call(called_func_id, function_lex_count, loop_lexical_id, i);
						if(called_func_name == "NULL_FUNC")
							insert_profile_request_next_identifier(i); //need function called via function-pointer to identify itself

						i++; //move to instruction just past call-site

						//insert profile-function-exit call after call-site
						insert_profile_function_exit_call(called_func_id, function_lex_count, loop_lexical_id, i);

						i--; //move up to the just inserted profile-exit-call, since loop-control increments

						function_lex_count++;
					}

					else if(TerminatorInst * termInst = dyn_cast<TerminatorInst>(i)) {
						update_dyn_instr_count(instr_count, i);
						instr_count = 0;
					}

					instr_count++;
				}
			}

			//insert profile-identifier at entry-point of function
			BasicBlock * entry = &(F.getEntryBlock());
			std::vector<Value*> Params;
			Params.push_back(ConstantInt::get(Type::getInt32Ty(F.getContext()), curr_func_id)); //turn on
			CallInst * prof_identifier = CallInst::Create(PROFILE_identifier, Params.begin(), Params.end(), "", entry->begin());

			if(curr_func_name == "main")
				create_profile_dumping_scope(F);

			return true;
		}

		bool isSpecial_PROFILE_function(const std::string& func_name) {
			if(func_name == "PROFILE_select_threadid" || func_name == "PROFILE_open" || func_name == "PROFILE_dump_setting"
				|| func_name == "PROFILE_function_entry" || func_name == "PROFILE_function_exit"
				|| func_name == "PROFILE_exception"
				|| func_name == "PROFILE_identifier" || func_name == "PROFILE_request_next_identifier"
			)
			{ return true; }

			return false;
		}

		void add_func_name_to_map(const std::string& func_name) {
			if(map_func_name_to_id.count(func_name) == 0) {
				map_func_name_to_id[func_name] = funcs_named_count;
				ofs_func_map << func_name << " " << funcs_named_count << std::endl;

				funcs_named_count++;
			}
		}

		CallInst * insert_profile_function_entry_call(int called_func_id, int function_lex_count, int loop_lexical_id, Instruction * i) {
			std::vector<Value*> Params;
			Params.push_back(ConstantInt::get(Type::getInt32Ty(i->getContext()), called_func_id));
			Params.push_back(ConstantInt::get(Type::getInt32Ty(i->getContext()), function_lex_count));
			Params.push_back(ConstantInt::get(Type::getInt32Ty(i->getContext()), loop_lexical_id));

			CallInst * prof_func_entry = CallInst::Create(PROFILE_function_entry, Params.begin(), Params.end(), "", i);
			return prof_func_entry;
		}

		CallInst * insert_profile_function_exit_call(int called_func_id, int function_lex_count, int loop_lexical_id, Instruction * i) {
			std::vector<Value*> Params2;
			Params2.push_back(ConstantInt::get(Type::getInt32Ty(i->getContext()), called_func_id));
			Params2.push_back(ConstantInt::get(Type::getInt32Ty(i->getContext()), function_lex_count));
			Params2.push_back(ConstantInt::get(Type::getInt32Ty(i->getContext()), loop_lexical_id));

			CallInst * prof_func_exit = CallInst::Create(PROFILE_function_exit, Params2.begin(), Params2.end(), "", i);
			return prof_func_exit;
		}

		CallInst * insert_profile_exception_call(int called_func_id, int function_lex_count, int loop_lexical_id, Instruction * i) {
			std::vector<Value*> Params;
			Params.push_back(ConstantInt::get(Type::getInt32Ty(i->getContext()), called_func_id));
			Params.push_back(ConstantInt::get(Type::getInt32Ty(i->getContext()), function_lex_count));
			Params.push_back(ConstantInt::get(Type::getInt32Ty(i->getContext()), loop_lexical_id));

			CallInst * prof_exception = CallInst::Create(PROFILE_exception, Params.begin(), Params.end(), "", i);
			return prof_exception;
		}

		CallInst * insert_profile_request_next_identifier(Instruction * i) {
			std::vector<Value*> Params;

			CallInst * prof_request_next_id = CallInst::Create(PROFILE_request_next_identifier, Params.begin(), Params.end(), "", i);
			return prof_request_next_id;
		}

		void create_profile_dumping_scope(Function &F) {
			BasicBlock * entry = &(F.getEntryBlock());
			std::vector<Value*> Params_ON;
			Params_ON.push_back(ConstantInt::get(Type::getInt32Ty(F.getContext()), 1)); //turn on
			CallInst * prof_dump_setting_ON = CallInst::Create(PROFILE_dump_setting, Params_ON.begin(), Params_ON.end(), "", entry->begin());


			std::vector<Value*> Params_OFF;
			Params_OFF.push_back(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0)); //turn off

			for (Function::iterator b = F.begin(), be = F.end(); b != be; b++) {
				//for each instruction in basic-block
				for(BasicBlock::iterator i=b->begin(), ie=b->end(); i != ie; i++) {
					if(ReturnInst * retInst = dyn_cast<ReturnInst>(i)) {
						CallInst * prof_dump_setting_OFF = CallInst::Create(PROFILE_dump_setting, Params_OFF.begin(), Params_OFF.end(), "", retInst);
					}
				}
			}
		}

		void setup_parms(Module &M) {
			std::ifstream ifs_func_map("func_map.dump");
			if(ifs_func_map) { //file exists and can be read
				std::string func_name;
				int func_id;
				while(ifs_func_map >> func_name >> func_id) {
					map_func_name_to_id[func_name] = func_id;
					if(func_id >= funcs_named_count)
						funcs_named_count = func_id + 1;
				}
			}
			ifs_func_map.close();

			ofs_func_map.open("func_map.dump", std::ios::app);
			if(!ofs_func_map) {
				std::cerr << "DVAProf::ERROR: Could not open 'func_map.dump' for writing" << std::endl;
				exit(1);
			}

			ofs_func_loop_hier.open("func_loop_hier.dump", std::ios::app);
			if(!ofs_func_loop_hier) {
				std::cerr << "DVAProf::ERROR: Could not open 'func_loop_hier.dump' for writing" << std::endl;
				exit(1);
			}

			dyn_instr_count_global = M.getNamedGlobal("dyn_instr_count");
			if(dyn_instr_count_global != 0) {
				std::cerr << "Found : "; errs() << *dyn_instr_count_global; std::cerr << std::endl;
			}
			else {
				std::cerr << "DVAProf::setup_parms(): ERROR: 'dyn_instr_count' not found" << std::endl;
				exit(1);
			}


			PROFILE_function_entry = M.getFunction("PROFILE_function_entry");
			if(PROFILE_function_entry == 0) {
				std::cerr << "DVAProf::setup_parms(): ERROR: PROFILE_function_entry() not found" << std::endl;
				exit(1);
			}

			PROFILE_function_exit = M.getFunction("PROFILE_function_exit");
			if(PROFILE_function_exit == 0) {
				std::cerr << "DVAProf::setup_parms(): ERROR: PROFILE_function_exit() not found" << std::endl;
				exit(1);
			}

			PROFILE_exception = M.getFunction("PROFILE_exception");
			if(PROFILE_exception == 0) {
				std::cerr << "DVAProf::setup_parms(): ERROR: PROFILE_exception() not found" << std::endl;
				exit(1);
			}

			PROFILE_dump_setting = M.getFunction("PROFILE_dump_setting");
			if(PROFILE_dump_setting == 0) {
				std::cerr << "DVAProf::setup_parms(): ERROR: PROFILE_dump_setting() not found" << std::endl;
				exit(1);
			}

			PROFILE_identifier = M.getFunction("PROFILE_identifier");
			if(PROFILE_identifier == 0) {
				std::cerr << "DVAProf::setup_parms(): ERROR: PROFILE_identifier() not found" << std::endl;
				exit(1);
			}

			PROFILE_request_next_identifier = M.getFunction("PROFILE_request_next_identifier");
			if(PROFILE_request_next_identifier == 0) {
				std::cerr << "DVAProf::setup_parms(): ERROR: PROFILE_request_next_identifier() not found" << std::endl;
				exit(1);
			}
		}
		
		void update_dyn_instr_count(int instr_count, Instruction * curr_instr) {
			if(instr_count == 0)
				return;

			Value * load_dyn = new LoadInst(dyn_instr_count_global, "load_dyn", curr_instr);
			ConstantInt * instr_count_val = ConstantInt::get(Type::getInt64Ty(curr_instr->getContext()), instr_count);
			Value * add_1 = BinaryOperator::Create(Instruction::Add, load_dyn, instr_count_val, "add", curr_instr);
			StoreInst * store_1 = new StoreInst(add_1, dyn_instr_count_global, curr_instr);
		}

		void construct_and_dump_func_loop_hier(
			LoopInfo& LI,
			std::vector<Loop *>& vLoops,
			std::map<Loop *, int>& map_loop_to_lexical_id
		)
		{
			ofs_func_loop_hier << "(0, [";

			bool bIsFirstIter = true;
			for(LoopInfo::iterator li=LI.begin(), le=LI.end(); li != le; li++) {
				int loop_lexical_id = vLoops.size();
				
				vLoops.push_back(*li);
				map_loop_to_lexical_id[*li] = loop_lexical_id;

				std::cerr << "TopLoop#" << loop_lexical_id << std::endl;

				if(bIsFirstIter)
					bIsFirstIter = false;
				else
					ofs_func_loop_hier << ", ";

				recursive_traverse_subloops(*li, vLoops, map_loop_to_lexical_id);
			}

			ofs_func_loop_hier << "])" << std::endl;
		}

		void recursive_traverse_subloops(
			Loop * loop,
			std::vector<Loop *>& vLoops,
			std::map<Loop *, int>& map_loop_to_lexical_id
		)
		{
			ofs_func_loop_hier << "(" << map_loop_to_lexical_id[loop] << ", [";

			bool bIsFirstIter = true;
			for(Loop::iterator li=loop->begin(), le=loop->end(); li != le; li++) {
				int loop_lexical_id = vLoops.size();

				vLoops.push_back(*li);
				map_loop_to_lexical_id[*li] = loop_lexical_id;

				std::cerr << "Loop#" << vLoops.size() << std::endl;

				if(bIsFirstIter)
					bIsFirstIter = false;
				else
					ofs_func_loop_hier << ", ";

				recursive_traverse_subloops(*li, vLoops, map_loop_to_lexical_id);
			}

			ofs_func_loop_hier << "])";
		}
		
	};

	//Global Definitions
	char DVAProf::ID = 0;
	RegisterPass<DVAProf> X("dva_profileinstrument", "Instrument application function for Dominant Variance Analysis");
}
