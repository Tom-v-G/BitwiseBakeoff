import json
import numpy as np
from geneticalgorithm import geneticalgorithm as ga




# print(data["ingredients"][0])

class CookieStuffs:

    def __init__(self, jsonfile, fitness_weights=None) -> None:
        # Open and read the JSON file
        with open(jsonfile, 'r') as file:
            self.ingredients = json.load(file)["ingredients"]

        self.beziercurve_dict = {
            "sugarcontent": [0, 2.165, 0.03, 0],
            "fatcontent": [0, 1.953, 0.468, 0],
            "price": [1, 0.982, 0.538, 0],
            "saltcontent": [0, 2.204, 0, 0],
            "watercontent": [0, 0.103, 2.1, 0],
            "flour": [0, 2.165, 0.03, 0]
        }
        
        if fitness_weights is not None:
            assert(len(fitness_weights) == len(self.beziercurve_dict))
            self.fitness_weights = fitness_weights
        else:
            self.fitness_weights = [1 for i in range(len(self.beziercurve_dict))]

    def normalize(self, ingredients, total_weight=300):
        # Normalise for batches of 300 gram cookie dough
        # [ 1 3 6 4 0 ]
        return (np.array(ingredients) * total_weight) / np.sum(ingredients)

    def calculate_content_percentage(self, normalized_ingredients, identifier):
        total_val = 0
        for idx, weight in enumerate(normalized_ingredients):
            total_val += self.ingredients[idx][identifier] * weight
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
        
        f1, f2, f3, f4, f5, f6 = self.fitness_weights
        
        fitness += f1 * self.bezier_curve(*self.beziercurve_dict["sugarcontent"], self.calculate_content_percentage(norm_ingredients, "sugarcontent"))
        fitness += f2 * self.bezier_curve(*self.beziercurve_dict["fatcontent"], self.calculate_content_percentage(norm_ingredients, "fatcontent"))
        fitness += f3 * self.bezier_curve(*self.beziercurve_dict["price"], self.calculate_content_percentage(norm_ingredients, "price"))
        fitness += f4 * self.exponential_decay(5, self.calculate_content_percentage(norm_ingredients, "saltcontent"))
        fitness += f5 * self.bezier_curve(*self.beziercurve_dict["watercontent"], self.calculate_content_percentage(norm_ingredients, "watercontent"))
        fitness += f6 * self.bezier_curve(*self.beziercurve_dict["flour"], norm_ingredients[2] / np.sum(norm_ingredients))

        return -1 * fitness

if __name__ == "__main__":
    fitness_weights = [3, 1, 0.2, 4, 1, 1]
    G = CookieStuffs("ingredients.json", fitness_weights=fitness_weights)
    recipe = [1, 3, 5, 6, 3, 1, 5, 1]
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
    print(f' \
    {best_recipe[0]} grams of water \n \
    {best_recipe[1]} grams of sugar \n \
    {best_recipe[2]} grams of flour \n \
    {best_recipe[3]} grams of eggs \n \
    {best_recipe[4]} grams of salt \n \
    {best_recipe[5]} grams of chocolate \n \
    {best_recipe[6]} grams of baking soda \n \
    {best_recipe[7]} grams of butter \n \
    ')
    # print(convergence)
    # print(solution)
    # GA
# {        
#          "name": "butter",
#          "type": "base",
#          "density": 0.911, 
#          "price" : 0.0118,
#          "liquid": false,
#          "sugarcontent": 0.001,
#          "fatcontent": 0.8,
#          "saltcontent": 0
# }