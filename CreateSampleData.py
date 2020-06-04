#!/usr/bin/env python
# coding: utf-8

# """
# Creating sample of good sentences by extracting sentences from Project Gutenberg. Incorrect samples will be read and copied from the toy data shared for Machine Translation.
# Creating a corpus of 5000 positive and 5000 negative sentences, positive means well formed or correctc english sentences and negative implies wrong sentences.
# """

from nltk.corpus import gutenberg
import re
import pandas as pd

fnames = gutenberg.fileids()
print(fnames)

# #### Replace 'austen_emma.txt' with any other name for generating training data differently.
POS_SAMPLE_FN = 'austen-emma.txt'
NEG_SAMPLE_FN = 'data\sample_mt_neg.tsv'
OUTPUT_FN = 'data\classification_data.tsv'

class createCorpus():
    def __init__(self,fn1, fn2, outputfn):
        self.pos_ex_fn = fn1
        self.neg_ex_fn = fn2
        self.outfn = outputfn
        self.corpus = self.extractCorpus()
    
    def extractCorpus(self):
        # .raw() returns raw text in a strign format
        raw_text = gutenberg.raw(self.pos_ex_fn)
         # print(raw_text[:500])
        
        #removing text inside []
        text = re.sub("^\[.*\]"," ", raw_text)
        #print("text after removing brackets ....")
#         print(text[:200])
        #removing VOLUME and Chapter nos.
        text = re.sub("\sVOLUME\s[A-Z]", " ", text)
#         print("removing volume....")
#         print(text[:500])
        text = re.sub("\sCHAPTER\s[A-Z]", " ", text)
        text = re.sub(r"--"," ",text)
        text = re.sub(r'\"'," ", text)
        #text = re.sub(r'[\"|\?\"|\.\"]'," ", text)
        text = re.sub(r'(?<=[MmSDsdr]){2}\.\s',' ',text)
        text = re.sub(r'(?<=[MmSDsdr]){3}\.\s',' ',text)
        text = re.sub(r'_.*_',' ',text)
        
        # removing  multiple spaces
        text = re.sub(r"\s+"," ",text)
        
        sents = re.split(r'\.|\?', text)
       # sents = text.lower().split(".")
#         print("sentences generated : ")
#        print(sents[1:10])
        return sents
    
    def getCorpus(self):
        return self.corpus
    
    def generatePositiveSentences(self):
        df = pd.DataFrame()
        df['sent'] = self.corpus
        df['class'] = 1
        return df
        
    def generateNegativeSentences(self):
        indf = pd.read_csv(self.neg_ex_fn, sep='\t', nrows=5000)
       # print(indf.info())

        df = pd.DataFrame()
        df['sent'] = indf['srctext']
        df['class'] = 0
#         print("Here ... df info", df.info())
        return df
    
    def writeFile(self):
        df1 = self.generatePositiveSentences()
        df2 = self.generateNegativeSentences()
        data = pd.concat([df1, df2])
        data.to_csv(self.outfn, sep='\t', index=False)
        print(data.info())
     

# In[ ]:
cobj = createCorpus(POS_SAMPLE_FN, NEG_SAMPLE_FN, OUTPUT_FN)
cobj.writeFile()