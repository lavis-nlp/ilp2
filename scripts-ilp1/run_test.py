import sys
from irt2_llm_prompter import model_prompter, test_logic, run_config

# Überprüfe, ob genügend Argumente übergeben wurden
if len(sys.argv) != 7:
    print(
        "Usage: python_script.py <model_path> <tensor_parallel_size> <system_prompt_path> <question_template_path> <dataset>"
    )
    sys.exit(1)

# Extrahiere die Argumente aus der Befehlszeile
dataset = sys.argv[1]
model_path = sys.argv[2]
tensor_parallel_size = int(sys.argv[3])
system_prompt_path = sys.argv[4]
question_template_path = sys.argv[5]
result_folder = sys.argv[6]

# Konfiguration erstellen
config = run_config.RunConfig.from_paths(
    question_template_path,
    system_prompt_path,
    dataset,
    model_path,
    tensor_parallel_size,
)

# Testlogik ausführen
test_logic.run_on_test(config, result_folder)
