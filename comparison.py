from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec 
from sklearn.metrics.pairwise import cosine_similarity
from bert_serving.client import BertClient


filename = "MTURK-771.csv"

for i in range(1):
    if i == 0:
        # Google word2vec
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    elif i == 1:
        # Glove
        #glove_input_file = 'glove.6B.200d.txt'
        word2vec_output_file = 'glove.6B.50d.txt.word2vec'
        #glove2word2vec(glove_input_file, word2vec_output_file)
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    elif i== 2:
        word2vec_output_file = 'glove.6B.100d.txt.word2vec'
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    elif i == 3:
        word2vec_output_file = 'glove.6B.200d.txt.word2vec'
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    elif i == 4:
        word2vec_output_file = 'glove.6B.300d.txt.word2vec'
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    elif i == 5:
        # FastText
        model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')    
    elif i == 6:
        model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec')

    bc = BertClient()
        
    dev = 0
    length = 0
    final = []
    summ = 0
    nf = 0
    with open(filename) as f:
        lis = [line.split() for line in f]   
         # create a list of lists
        length = len(lis)
        for i, x in enumerate(lis):              #print the list items 
            if (i == 0):
                continue
            t = x[0].split(',')
            #t = x # For space separated files
            a = t[0]
            b = t[1]
            c = t[2]
            c = float(c)*2
            summ += float(c)
            try:
                res = model.similarity(a,b)*10
            except KeyError:
                res = float(c)
                nf = nf + 1
            dev = dev + abs(res - float(c))
    dev = dev/length        
    final.append(dev)
    summ = summ/length
    percentage = (dev/summ)*100
    print (dev)
    print(percentage)
    print(nf)
    
    
    
# BERT    
bc = BertClient()   
dev = 0
length = 0
final = []
summ = 0
nf = 0
with open(filename) as f:
    lis = [line.split() for line in f]   
     # create a list of lists
    length = len(lis)
    for i, x in enumerate(lis):              #print the list items 
        if (i == 0):
            continue
        t = x[0].split(',')
        #t = x
        a = t[0]
        b = t[1]
        c = t[2]
        c = float(c)*2
        summ += float(c)
        try:
            #res = model.similarity(a,b)*10
            x = bc.encode([a, b])
            t1 = x[0]
            t2 = x[1]
            t1 = t1.reshape(1,768)
            t2 = t2.reshape(1,768)
            res = cosine_similarity(t1,t2)
            res = res[0][0]*10
        except KeyError:
            res = float(c)
            nf = nf + 1
        dev = dev + abs(res - float(c))
dev = dev/length        
final.append(dev)
summ = summ/length
percentage = (dev/summ)*100
print(dev)
print(percentage)
print(nf)    