import json
import numpy as np
from geneticalgorithm import geneticalgorithm as ga


class CookieStuffs:

    def __init__(self, ingredients, fitness_weights=None, algorithm_params=None) -> None:
        self.ingredients = ingredients
        self.beziercurve_dict = {
            "sugarcontent": [0, 2.165, 0.03, 0],
            "fatcontent": [0, 1.953, 0.468, 0],
            "price": [1, 0.982, 0.538, 0],
            "saltcontent": [0, 2.204, 0, 0],
            "watercontent": [0, 0.103, 2.1, 0],
            "flour": [0, 2.165, 0.03, 0],
            "flavor_enhancer_weight": [-25]
        }

        if algorithm_params is not None: 
            self.algorithm_params = algorithm_params 
        else:    
            self.algorithm_params =  {
                   'max_num_iteration': 3000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None
            }
        
        if fitness_weights is not None:
            assert(len(fitness_weights) == len(self.beziercurve_dict))
            self.fitness_weights = fitness_weights
        else:
            self.fitness_weights = [1 for i in range(len(self.beziercurve_dict))]

    def normalize(self, ingredients, total_weight=300):
        # Normalise for batches of 300 gram cookie dough
        return (np.array(ingredients) * total_weight) / np.sum(ingredients)

    def calculate_content_percentage(self, normalized_ingredients, identifier, type_filter=None):
        total_val = 0
        for idx, weight in enumerate(normalized_ingredients):
            if type_filter is None:
                total_val += self.ingredients[idx][identifier] * weight
            else:
                if self.ingredients[idx]["type"] == type_filter:
                    total_val +=  weight
        return total_val / np.sum(normalized_ingredients)
    
    @staticmethod
    def bezier_curve(y0, y1, y2, y3, t):
        #x = (1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * ((1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3))
        y = (1 - t) * ((1 - t) * ((1 - t) * y0 + t * y1) + t * ((1 - t) * y1 + t * y2)) + t * ((1 - t) * ((1 - t) * y1 + t * y2) + t * ((1 - t) * y2 + t * y3))
        return y
    
    @staticmethod
    def exponential_decay(c, t):
        return np.exp(-c * t)

    def fitness_function(self, norm_ingredients):
        fitness = 0
        
        f1, f2, f3, f4, f5, f6, f7 = self.fitness_weights
        
        fitness += f1 * self.bezier_curve(*self.beziercurve_dict["sugarcontent"], self.calculate_content_percentage(norm_ingredients, "sugarcontent"))
        fitness += f2 * self.bezier_curve(*self.beziercurve_dict["fatcontent"], self.calculate_content_percentage(norm_ingredients, "fatcontent"))
        fitness += f3 * self.bezier_curve(*self.beziercurve_dict["price"], self.calculate_content_percentage(norm_ingredients, "price"))
        fitness += f4 * self.exponential_decay(5, self.calculate_content_percentage(norm_ingredients, "saltcontent"))
        fitness += f5 * self.bezier_curve(*self.beziercurve_dict["watercontent"], self.calculate_content_percentage(norm_ingredients, "watercontent"))
        fitness += f6 * self.bezier_curve(*self.beziercurve_dict["flour"], norm_ingredients[2] / np.sum(norm_ingredients))
        fitness += f7 * self.exponential_decay(*self.beziercurve_dict["flavor_enhancer_weight"], self.calculate_content_percentage(norm_ingredients, None, type_filter="flavor enhancer"))

        return -1 * fitness

    def GA(self, total_weight=300):
        
        varbound=np.array([[1, (total_weight / 2)]]*len(self.ingredients))

        model=ga(function=self.fitness_function, 
             dimension=len(self.ingredients),
             variable_type='int',
             variable_boundaries=varbound,
             algorithm_parameters=self.algorithm_params,
             convergence_curve=False)
        
        model.run()
        solution=model.output_dict

        return self.normalize(solution['variable'])

if __name__ == "__main__":

    fitness_weights = [3, 1, 0.2, 4, 1, 1, 1]
    
    # Open and read the JSON file
    with open("ingredients.json", 'r') as file:
        ingredients = json.load(file)["ingredients"]

    G = CookieStuffs(ingredients, fitness_weights=fitness_weights)
    recipe = [1, 3, 5, 6, 3, 1, 5, 1, 1, 5]
    print(np.sum(recipe))

    total_weight = 300
    norm = G.normalize(recipe, total_weight=total_weight)
    print(norm)
    # print(G.calculate_content_percentage(norm, "liquid"))
    print(G.fitness_function(norm))
    
    varbound=np.array([[1, (total_weight / 2)]]*len(recipe))

    algorithm_param = {
                   'max_num_iteration': 3000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

    model=ga(function=G.fitness_function, 
             dimension=len(recipe),
             variable_type='int',
             variable_boundaries=varbound,
             algorithm_parameters=algorithm_param,
             convergence_curve=False)

    model.run()
    convergence=model.report
    solution=model.output_dict

    #print(solution.variable)
    best_recipe = G.normalize(solution['variable'])
    
    for idx, weight in enumerate(best_recipe):
        print(f"{weight} grams of {ingredients[idx]['name']}\n")

