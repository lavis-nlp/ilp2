from vllm import LLM, SamplingParams
from irt2_llm_prompter import model_prompter, test_logic, run_config
sampling_params = SamplingParams(temperature=0, top_p=1,use_beam_search=True,best_of=2,max_tokens=512)
model = model_prompter.Model("/data/tyler/llms/llama3/Meta-Llama-3-70B-Instruct/",sampling_params,4)
model.load_model()
config = run_config.RunConfig.from_paths("prompts/question/prompt_templates_tiny_v2.json","prompts/system/sysp_generic_to_json_v8.json","/data/tyler/llms/irt2/data/irt2/irt2-cde-miniscule-fixed/")

test_logic.run_test(config,model)
