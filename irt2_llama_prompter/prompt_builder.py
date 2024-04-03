from enum import Enum

class PromptType(Enum):
    GENERIC = "Generischer Prompt"
    SPECIFIC = ""

specific_prompts = {
    "profession": "Profession Prompt"
}

class prompt_builder():
    def __init__(self):
        self.promptType = PromptType.GENERIC
    
    def setType(self, type):
        self.promptType = type

    def setMention(self, mention):
        self.mention = mention

    def setRelation(self, relation):
        self.relation = relation

    def buildPrompt(self):
        prompt = ""
        if self.promptType == PromptType.GENERIC:
            prompt += self.promptType.value
        elif self.promptType == PromptType.SPECIFIC:
            prompt += specific_prompts.get(self.relation)
        return prompt
    
prompt_builder = prompt_builder()
prompt_builder.setMention("Trump")
prompt_builder.setRelation("profession")
prompt_builder.setType(PromptType.SPECIFIC)    
print(prompt_builder.buildPrompt())