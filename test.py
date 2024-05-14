from irt2_llm_prompter import model_prompter, test_logic, run_config
config = run_config.RunConfig.from_paths("prompts/question/prompt_templates_large_v4.json","prompts/system/sysp_generic_to_json_v8.json",'irt2-cde-large','/data/hiwi/lukas/llms/llama3/Meta-Llama-3-70B-Instruct/',4)
test_logic.run_on_test(config)
