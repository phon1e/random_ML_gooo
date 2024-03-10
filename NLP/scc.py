import csv 
import numpy as np
from nltk import word_tokenize as tokenize

class question:
    
    def __init__(self,aline):
        self.fields=aline
    
    def get_field(self,field):
        return self.fields[question.colnames[field]]
    
    def add_answer(self,fields):
        self.answer=fields[1]
   
    def chooseA(self):
        choices = ["a","b","c","d","e"]
        return np.random.choice(choices)
    
    # adding choose by unigram model
    def choosesunigram(self, lm):
        choices = ["a","b","c","d","e"]
        probs = [lm.unigram.get(self.get_field(ch+ ")"), 0) for ch in choices]
        max_prob = max(probs)
        bestchoices = [ch for ch, prob in zip(choices, probs) if prob == max_prob]
        
        if bestchoices >1:
            bestchoices = np.random.choices(bestchoices)
        return bestchoices

    #add context 
    def get_tokens(self):
        return ["START"]+tokenize(self.fields[question.colnames["question"]]) + ["__END"]
    
    def get_left_context(self, window= 1, target="_____"):
        found = -1
        sent_tokens = self.get_tokens()
        for i, token in enumerate(sent_tokens):
            if token == target:
                found = i
                break
        if found>-1:
            return sent_tokens[i-window: i]
        else:
            return []
        
    def get_right_context(self, window=1, target="_____"):
        found = -1
        sent_tokens = self.get_tokens()
        for i, token in enumerate(sent_tokens):
            if token == target:
                found = i
                break
        if found>-1:
            return sent_tokens[found+1: found+window+1]
        else:
            return []
    
    def choose(self,lm,method="bigram",choices=[]):
        if choices==[]:
            choices=["a","b","c","d","e"]
        if method=="bigram":
            rc=self.get_right_context(window=1)
            lc=self.get_left_context(window=1)
            probs=[lm.get_prob(rc[0],[self.get_field(ch+")")],methodparams={"method":method.split("_")[0]})*lm.get_prob(self.get_field(ch+")"),lc,methodparams={"method":method.split("_")[0]}) for ch in choices]
        elif method=="bigram_right":
            context=self.get_right_context(window=1)
            probs=[lm.get_prob(context[0],[self.get_field(ch+")")],methodparams={"method":method.split("_")[0]}) for ch in choices]
        else:
            #this covers bigram_left and unigram
            context=self.get_left_context(window=1)
            probs=[lm.get_prob(self.get_field(ch+")"),context,methodparams={"method":method.split("_")[0]}) for ch in choices]
        maxprob=max(probs)
        bestchoices=[ch for ch,prob in zip(choices,probs) if prob == maxprob]
        #if len(bestchoices)>1:
        #    print("Randomly choosing from {}".format(len(bestchoices)))
        return np.random.choice(bestchoices)
    
    #choose back off 
    def choose_backoff(self, lm, methods=['bigram', 'unigram'], choices=["a","b","c","d","e"]):
        context = self.get_left_context(window = 1)
        probs=[lm.get_prob(self.get_field(ch+")"),context,methodparams={"method":methods[0]}) for ch in choices]
        maxprob=max(probs)
        bestchoices=[ch for ch,prob in zip(choices,probs) if prob == maxprob]
        if len(bestchoices)>1:
            print("Backing off on {}".format(len(bestchoices)))
        return self.choose(lm,choices=bestchoices,method=methods[1])


    def predict(self,method="chooseA",model=mylm):
        if method=="chooseA":
            return self.chooseA()
        elif method=="random":
            return self.chooserandom()
        elif method=="bigram_backoff":
            return self.choose_backoff(mylm,methods=["bigram","unigram"])
        else:
            return self.choose(mylm,method=method)
        
    def predict_and_score(self,method="chooseA"):
        
        #compare prediction according to method with the correct answer
        #return 1 or 0 accordingly
        prediction=self.predict(method=method)
        if prediction ==self.answer:
            return 1
        else:
            return 0


class scc_reader:
    
    def __init__(self,qs=questions,ans=answers):
        self.qs=qs
        self.ans=ans
        self.read_files()
        
    def read_files(self):
        
        #read in the question file
        with open(self.qs) as instream:
            csvreader=csv.reader(instream)
            qlines=list(csvreader)
        
        #store the column names as a reverse index so they can be used to reference parts of the question
        question.colnames={item:i for i,item in enumerate(qlines[0])}
        
        #create a question instance for each line of the file (other than heading line)
        self.questions=[question(qline) for qline in qlines[1:]]
        
        #read in the answer file
        with open(self.ans) as instream:
            csvreader=csv.reader(instream)
            alines=list(csvreader)
            
        #add answers to questions so predictions can be checked    
        for q,aline in zip(self.questions,alines[1:]):
            q.add_answer(aline)
        
    def get_field(self,field):
        return [q.get_field(field) for q in self.questions] 
    
    def predict(self,method="chooseA"):
        return [q.predict(method=method) for q in self.questions]
    
    def predict_and_score(self,method="chooseA"):
        scores=[q.predict_and_score(method=method) for q in self.questions]
        return sum(scores)/len(scores)
    