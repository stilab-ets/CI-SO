from GA.optimizer import Optimizer

def get_average_score(pop):
    """Get the average score for a group of solutions."""
    total_scores = 0
    for solution in pop:
        total_scores += solution.score
    return total_scores / len(pop)

"""Generate the optimal params with the genetic algorithm."""
""" Args:
        GA_params: Params for GA
        all_possible_params (dict): Parameter choices for the model
"""
def generate(all_possible_params):
   
    GA_params = {
            "population_size": 2,
            "max_generations":2,
            "retain": 0.7,
            "random_select":0.1,
            "mutate_chance":0.1
            }
    
    print("params of GA" , GA_params)
    optimizer = Optimizer(GA_params ,all_possible_params)
    pop = optimizer.create_population(GA_params['population_size'])
    # Evolve the generation.
    for i in range(GA_params['max_generations']):
        print("*********************************** REP(GA) ",(i+1))
        # Get the average score for this generation.
        average_accuracy = get_average_score(pop)
        # Print out the average accuracy each generation.
        print("Generation average: %.2f%%" % (average_accuracy * 100))
        # Evolve, except on the last iteration.
        if i != (GA_params['max_generations']):
            print("Generation evolving..")
            evolved = optimizer.evolve(pop)
            if(len(evolved)!=0):
                pop=evolved
        else:
            pop = sorted(pop, key=lambda x: x.score, reverse=True)
    # Print out the top 2 solutions.
    size = len(pop)
    if size < 3:
        print_pop(pop[:size])
    else:
        print_pop(pop[:3])
    return pop[0].params ,pop[0].model

def print_pop(pop):
    for solution in pop:
        solution.print_solution()    
