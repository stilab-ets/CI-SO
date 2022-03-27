"""Class that represents the solution to be evolved."""
import random,gensim
from gensim.models import CoherenceModel
class Solution():
    def __init__(self, all_possible_params):
        self.score = None
        self.all_possible_params = all_possible_params
        self.params = {}  #  represents model parameters to be picked by creat_random method
        self.model = None
        
        
    def set_compute_score(self):
        # Compute Coherence Score
        lda_model = gensim.models.LdaMulticore(**self.params)
        coherence_model_lda = CoherenceModel(
                                             model=lda_model,
                                             texts=self.params["corpus"], 
                                             dictionary=self.params["id2word"],
                                             coherence='c_v'
                                             )
        coherence_lda = coherence_model_lda.get_coherence()
        self.score =  coherence_lda
        self.model = lda_model
        
    """Create the model random params."""
    def create_random(self):
        for key in self.all_possible_params:
            if (key not in "id2word" and key not in "corpus"):
                self.params[key] = random.choice(self.all_possible_params[key])
            else:
                self.params[key] = self.all_possible_params[key]
        
        self.set_compute_score()


    
        
        
    def set_params(self, params):
        self.params = params
        self.set_compute_score()
        
      
            
    """Print out a network."""
    def print_solution(self):
        print("for params ", self.params , "the score in the train = ",self.score)