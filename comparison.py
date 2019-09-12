from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec 


# Load vectors directly from the file
# Google word2vec model
# =============================================================================
#model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# =============================================================================
# Glove model
# =============================================================================
# glove_input_file = 'glove.6B.100d.txt'
# word2vec_output_file = 'glove.6B.100d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)
# model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# =============================================================================

#FastText
#1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset
# =============================================================================
#model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
# =============================================================================


for i in range(3):
    if i== 0:
        # Google word2vec
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    elif i == 1:
        # Glove
        glove_input_file = 'glove.6B.100d.txt'
        word2vec_output_file = 'glove.6B.100d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    elif i == 2:
        # FastText
        model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')

        
    dev = 0
    length = 0
    final = []
    summ = 0
    with open("wordsim353/ws353.csv") as f:
        lis = [line.split() for line in f]   
         # create a list of lists
        length = len(lis)
        for i, x in enumerate(lis):              #print the list items 
            if (i == 0):
                continue
            t = x[0].split(',')
            a = t[0]
            b = t[1]
            c = t[2]
            summ += float(c)
            try:
                res = model.similarity(a,b)*10
            except KeyError:
                res = float(c)
            dev = dev + abs(res - float(c))
    dev = dev/length        
    final.append(dev)
    summ = summ/length
    percentage = (dev/summ)*100
    print (dev)
    print(percentage)