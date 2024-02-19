import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Tokenizes a sentence into individual words.
    
    Parameters:
        sentence (str): The input sentence to be tokenized.
        
    Returns:
        list: A list of tokens extracted from the input sentence.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stems a word to its root/base form.
    
    Parameters:
        word (str): The word to be stemmed.
        
    Returns:
        str: The stemmed form of the input word.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Creates a bag-of-words representation of a tokenized sentence.
    
    Parameters:
        tokenized_sentence (list): A list of tokens representing a sentence.
        all_words (list): A list of all unique words in the dataset.
        
    Returns:
        numpy.ndarray: A numpy array representing the bag-of-words vector for the input sentence.
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
