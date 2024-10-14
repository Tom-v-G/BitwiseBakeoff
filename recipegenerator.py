import json
import numpy as np




# print(data["ingredients"][0])

class CookieStuffs:

    def __init__(self, jsonfile) -> None:
        # Open and read the JSON file
        with open(jsonfile, 'r') as file:
            self.ingredients = json.load(file)["ingredients"]

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
    
    def fitness_function(self, norm_ingredients):
        beziercurve_dict = {
            "sugar": [0.002, 1.503, 0.038, -0.001]
        }

        return self.bezier_curve(*beziercurve_dict["sugar"], self.calculate_content_percentage(norm_ingredients, "sugarcontent"))

        # Low sugar content

        # Price

        # proper mixture liquid / solid

        # Fat content

        # Salt content

        return

if __name__ == "__main__":
    G = CookieStuffs("ingredients.json")
    recipe = [1, 3, 5, 6, 3, 1, 5, 1]
    print(np.sum(recipe))
    norm = G.normalize(recipe)
    print(norm)
    print(G.calculate_content_percentage(norm, "liquid"))
    print(G.fitness_function(norm))
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