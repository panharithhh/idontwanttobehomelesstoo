
def get_stats(ids):
    # The goal is
    counts = {}

    for k,v in ids.items():
        for pair in list(zip(k, k[1:])):
            counts[pair] = counts.get(pair,0) + v

   
    return counts

def get_unique_pretoken(text):

    word_counts = {} 

    for word in text:
        word_bytes = word.encode("utf-8") 
        word_counts[word_bytes] = word_counts.get(word_bytes,0) +1

    return word_counts 

class BpeTokenizer():

    def __init__():
        pass


    def train(self,):
        pass


    def encdoe(self,):
        pass


    def decode(self,):
        pass
