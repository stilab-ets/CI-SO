# Importing modules
import pandas as pd
import re# Load the regular expression library
import string
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import spacy#pip install spacy
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) 
#nlp = spacy.load('en', disable=['parser', 'ner'])
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

from gensim.models import TfidfModel
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import nltk
import gensim.corpora as corpora
from pprint import pprint
import re# Load the regular expression library
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
import os
import string
import spacy#pip install spacy
import warnings
warnings.filterwarnings("ignore")

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import threading
from tqdm import tqdm
import threading
from gensim.models import CoherenceModel
import pandas as pd
import re# Load the regular expression library
import string
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import spacy#pip install spacy
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) 
#nlp = spacy.load('en', disable=['parser', 'ner'])
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

from gensim.models import TfidfModel

def sent_to_words(sentences):
    for sentence in sentences:
        #removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        

def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]  
                 
def train_lda_thread(item,i,corpus,id2word):
   
    item["lda_model"] = gensim.models.LdaMulticore(**item["params"])
    print("lda_model ", i," trained")
    item["number"] =i

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ','VERB', 'ADV']):#
    """https://spacy.io/api/annotation"""
    #docs = pd.Series( (" ".join(v) for v in texts) )
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        #for  token in doc:
            #if token.pos_ in allowed_postags:
                #print(token.lower_, token.pos_, token.lemma_ )
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
    
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
    
 def post_processing(LDA_models, best_item, corpus, id2word ,data_lemmatized, Posts):
    
    
    optimal_model = best_item["lda_model"]
    #print(optimal_model.show_topics(formatted=False))
    pprint(optimal_model.print_topics(num_words=20))

    
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=optimal_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    ############ Finding the dominant topic in each sentence
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts = Posts)
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    # Show
    df_dominant_topic.head(10)
    
    ################# Find the most representative document for each topic
    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()
    
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                 grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                                axis=0)
    
    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    
    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    
    # Show
    sent_topics_sorteddf_mallet.head()
    
    #################### Topic distribution across documents
    # Number of Documents for Each Topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    
    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)
    
    # Topic Number and Keywords
    topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
    
    # Concatenate Column wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
    
    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
    
    # Show
    df_dominant_topics   

#############################  DATA PREPROCESSING ##########################

#data = pd.read_excel('C:/Users/AQ01490/OneDrive - ETS/9. AUT2021/ci-in-stackoverflow/Analysis & results/CI posts.xlsx')
def preprocess(data , column=""):
    
    #1. remove code snippets, URLs and HTML tags
    titles = data['Title'].values 
    body =  data['Body'].values  
    fn = lambda idx: str(titles[idx])+str(body[idx])
    
  
    
    Posts = pd.Series([fn(idx) for idx in range(len(body))])
    #Posts =  data['Title']
    Posts =  Posts.map(lambda x: re.sub('continuous integration', '', x))
    Posts =  Posts.map(lambda x: re.sub('<code>.+?</code>', '', x))#remove code snippets
    Posts =  Posts.map(lambda x: re.sub('<a .+?</a>', '', x))#remove links
    Posts =  Posts.map(lambda x: re.sub('<img .+?>', '', x))#remove links
    Posts =  Posts.map(lambda x: re.sub('<.+?>', '', x))#remove HTML tags
    ############2. We remove the English stop words, numbers, punctuation marks and other non-alphabetic characters
    
    #Posts =  Posts.map(lambda x: re.sub('[0-9]', '', x))#numbers
    #Posts =  Posts.map(lambda x: '' if x.isdigit() else x)#numbers
    Posts =  Posts.map(lambda x: re.sub(r'[' + string.punctuation + ']', '', x))# Replace all special characters
    
    Posts =  Posts.map(lambda x: x.lower())
    Posts =  Posts.map(lambda x: re.sub('testing', 'test', x))
    Posts =  Posts.map(lambda x: re.sub('building', 'build', x))
    data_words = list(sent_to_words(Posts))
   
    # Remove Stop Words
    data_words = remove_stopwords(data_words , stop_words)
    #remove extra words
    extra = ["ci","cicd","probably","one", "even","every","another","whole","thats","sad","question","answer","please","everything"]
    data_words = remove_stopwords(data_words , extra)
   
    #************************##### bigrams and triagrams
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    #trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    #trigram_mod = gensim.models.phrases.Phraser(trigram)
     # See trigram example
     
    # Define functions for bigrams
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    data_words_bigrams = make_bigrams(data_words)
    
     # Do lemmatization keeping only noun, adj, vb, adv
     data_lemmatized = lemmatization(data_words_bigrams,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
     
     
    
    
     # # # Create Corpus
     id2word = corpora.Dictionary(data_lemmatized)
     texts = data_lemmatized
     corpus = [id2word.doc2bow(text) for text in texts]
     # # Human readable format of corpus (term-frequency)
     print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
    
    return corpus,id2word,data_lemmatized,Posts
     
 def run_tuning(data_lemmatized,id2word):
   import  GA.GARunner as GARunner
   search_params = {
       "corpus":data_lemmatized,  
       "id2word":id2word,
      'num_topics'   : [range(2, 50, 1)]# Number of topics, 25, 30,35,40,15,20,25
      ,'iterations'       : [range(10, 5000, 100)]# Max learning iterations
      ,'chunksize '       : [range(10, 2000, 100)]# Number of documents to be used in each training chunk
       ,'passes '       : [range(1, 100, 1)]# Number of passes through the corpus during training
    }

            
    best_params ,best_model = GARunner.generate(search_params)

    
    





            