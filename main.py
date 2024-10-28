import torch
import json
from datetime import datetime as time

from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from ingredientsgenerator import CookieStuffs


class LLM:
    template_messages = [
        SystemMessage(content='''
                    You are an assistant that generates chocolate chip cookie recipes. You are given a list of ingredients. Output only a recipe to bake chocolate chip cookies and an ingredients list at the top. Start with the title. The title should be based on the included note. 
                    '''),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]

    model = Ollama(model="llama3:latest")
    prompt_template = ChatPromptTemplate.from_messages(template_messages)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    runnable = (
        {"text": RunnablePassthrough()} | prompt_template | model | StrOutputParser()
    )

    def generate_recipe(self, ingredients_qtt, ingredients_dict, note):
        
        ingredients_string = f'Note: {note}\n'
        for idx, ingredient in enumerate(ingredients_qtt):
            ingredients_string += f'{ingredient:.2f} grams of {ingredients_dict[idx]["name"]}'
            if idx != (len(ingredients_qtt) -1): ingredients_string += '\n'
        return self.runnable.invoke(ingredients_string)

# Open and read the JSON file
def read_ingredients_from_json(jsonfile):        
    with open(jsonfile, 'r') as file:
        ingredients_dict = json.load(file)["ingredients"]
    return ingredients_dict

if __name__ == "__main__":
    ingredients_dict = read_ingredients_from_json("ingredients.json")
    fitness_weights_dict = {
        'low cost cookie': [1, 1, 5, 1, 1, 1, 1],
        'sweet cookie': [5, 1, 1, 1, 1, 1, 1],
        'chocolaty cookie': [1, 1, 0.2, 1, 1, 1, 1],
        'non fat cookie': [1, 5, 1, 1, 1, 1, 1]
                            }
    notes = list(fitness_weights_dict.keys())
    for idx, fitness_weights in enumerate(fitness_weights_dict.values()):
        G = CookieStuffs(ingredients_dict, fitness_weights=fitness_weights)

        ingredients = G.GA(300)

        llm = LLM()
        recipe = llm.generate_recipe(ingredients, ingredients_dict, notes[idx])
        print(recipe)

        with open(f'recipes/recipe_{idx}', 'w') as f:
            f.write(recipe)