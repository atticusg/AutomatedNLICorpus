import random
import os
import copy
import pickle
from nltk.sem.logic import *
from nltk.inference import *
from nltk import Prover9
from joblib import Parallel, delayed

class sentence:
    def __init__(self, core, passive, negate, adverb, adjective1, adjective2, determiners):
        self.core = core
        self.passive = passive
        self.negate = negate
        self.determiners = determiners
        self.negation = ["", ""]
        self.adverb = adverb 
        self.adjective1 = adjective1
        self.adjective2 = adjective2
        self.adverb_position = ""
        self.negate_adverb = ""
        self.verb_adverb = ""
        self.sentence_adverb = ""
        if random.randint(0,1) and self.negate and self.adverb != "":
            self.negate_adverb = self.adverb
            self.adverb_position = "before negation"
            self.sentence_adverb = ""
            self.verb_adverb = ""
        elif random.randint(0,1) and self.adverb != "":
            self.verb_adverb = self.adverb
            self.adverb_position= "before verb"
        elif self.adverb != "":
            self.sentence_adverb = self.adverb
            self.adverb_position = "end"
        self.non_passive_verb_index = 0
        if negate:
            self.negation = ["not", "does not"]
            self.non_passive_verb_index = 2
        if self.passive:
            temp = self.adjective1
            self.adjective1 = self.adjective2#This ensures that identical adjectives appearing in
            self.adjective2 = temp # both the premise and hypothesis apply to the same noun
            self.final = self.construct_string([self.determiners[0],self.adjective1,self.core[2],"is",self.negate_adverb,self.negation[0],self.verb_adverb,self.core[1][1],"by",self.determiners[1],self.adjective2,self.core[0],self.sentence_adverb])
        else:
            self.final = self.construct_string([self.determiners[0],self.adjective1,self.core[0],self.negate_adverb,self.negation[1],self.verb_adverb,self.core[1][self.non_passive_verb_index],self.determiners[1],self.adjective2,self.core[2],self.sentence_adverb])
        self.change_an()
        self.construct_logical_form()
        #print(self.final)
        #print(self.logical_form)

    def change_an(self):
        if self.final[0] == "a" and self.final[2] in "aeiou":
            self.final = "an" + self.final[1:]
        for i in range(len(self.final) - 2):
            if self.final[i:i+3] == " a " and self.final[i+3] in "aeiou":
                self.final = self.final[:i+2] + "n" + self.final[i+2:]
                

    def construct_string(self,lst):
        result = ""
        for word in lst:
            if word != "":
                result += word + " "
        return result[:-1]

    
#    def construct_logical_form(self):
#        logical_form = ""
#        working_core = list(copy.copy(self.core))
#        verb = working_core[1][1]
#        if self.passive:
#           working_core.reverse()
#        logical_form = verb + "(x,y)"
#        if self.adverb_position == "before verb" or self.adverb_position == "end":
#            logical_form = "("+logical_form+"&"+self.adverb+verb +"(x,y))"
#        temp = working_core[2] + "(y)"
#        if self.adjective2 != "":
#            temp = temp + "&" + self.adjective2+working_core[2] + "(y)"
#        if self.determiners[1] == "every":
#            logical_form = "all y.(" + temp + "->" + logical_form + ")"
#        elif self.determiners[1] == "a":
#           logical_form = "exists y.(" + temp + "&" + logical_form + ")"
#        else:
#            logical_form = "exists y.((" + temp + "&" + logical_form + ")&the(y))"
#        if self.negate:
#            logical_form = "-" + logical_form
#        if self.adverb_position == "before negation":  
#            logical_form = "(" + logical_form + "&" + self.adverb+"not"+verb+ ")"
#        temp = working_core[0] + "(x)"
#        if self.adjective1 != "":
#            temp = "(" + temp + "&" + self.adjective1 + working_core[0] + "(x))"
#        if self.determiners[0] == "every":
#            logical_form = "(all x.(" + temp + "->" + logical_form + "))"
#        elif self.determiners[0] == "a":
#            logical_form = "(exists x.(" + temp + "&" + logical_form + "))"
#        else:
#            logical_form = "(exists x.((" + temp + "&" + logical_form + ")&the(x)))"
#        self.logical_form = logical_form 

    def construct_logical_form(self):
        logical_form = ""
        append_formula = []
        working_core = list(copy.copy(self.core))
        verb = working_core[1][0]
        constant1 = "(constant1)"
        constant2 = "(constant2)"
        if self.passive:
            constant1 = "(constant2)"
            constant2 = "(constant1)"
        first_noun_arg = "(y)"
        second_noun_arg = "(x)"
        if self.determiners[1] == "the":
            first_noun_arg = constant1
        if self.determiners[0] == "the":
            second_noun_arg = constant2
        verb_arg = "(" + second_noun_arg[1:-1] + ","+ first_noun_arg[1:-1] + ")"
        if self.passive:
            working_core.reverse()
            verb_arg = "(" + first_noun_arg[1:-1] + ","+ second_noun_arg[1:-1] + ")"
        logical_form = verb + verb_arg
        if self.adverb_position == "before verb" or self.adverb_position == "end":
            logical_form = self.adverb + logical_form
            append_formula.append("all y.all x.(" + logical_form +"->" + verb + verb_arg + ")")
        temp = working_core[2] + first_noun_arg
        if self.adjective2 != "":
            temp = self.adjective2+working_core[2] + first_noun_arg
            append_formula.append("all x.(" + self.adjective2+working_core[2] + "(x)" +"->" + working_core[2] + "(x)" + ")")
        if self.determiners[1] == "every":
            logical_form = "all y.(" + temp + "->" + logical_form + ")"
        elif self.determiners[1] == "a":
            logical_form = "exists y.(" + temp + "&" + logical_form + ")"
        else:
            logical_form = "(" + temp + "&" + logical_form + ")"
        if self.negate:
            logical_form = "-" + logical_form
        if self.adverb_position == "before negation":  
            logical_form = "(" + logical_form + "&" + self.adverb+"not"+verb+ verb_arg + ")"
        temp = working_core[0] + second_noun_arg
        if self.adjective1 != "":
            temp = self.adjective1 + working_core[0] +second_noun_arg
            append_formula.append("all x.(" + self.adjective1+working_core[0] + "(x)" +"->" + working_core[0] + "(x)" + ")")
        if self.determiners[0] == "every":
            logical_form = "all x.(" + temp + "->" + logical_form + ")"
        elif self.determiners[0] == "a":
            logical_form = "exists x.(" + temp + "&" + logical_form + ")"
        else:
            logical_form = "(" + temp + "&" + logical_form + ")"
        for form in append_formula:
            logical_form = logical_form + "&" + form
        self.logical_form = logical_form 


def process_data():
    train = dict()
    val = dict()
    test = dict()
    ratio = 0.6
    categories = ["agents", "transitive_verbs", "things", "determiners", "adverbs", "adjectives1","adjectives2"]
    for c in categories:
        with open(os.path.join("data", c + ".txt"),"r") as f:
            stuff = f.readlines()
            if c != "transitive_verbs":
                stuff = [_.strip() for _ in stuff]
            else:
                stuff = [_.strip().split() for _ in stuff]
        random.shuffle(stuff)
        if c != "determiners":
            train[c] = stuff[:int(len(stuff)*ratio)]
            val[c] = stuff[int(len(stuff)*ratio):int(len(stuff)*(ratio+(1-ratio)*0.5))]
            test[c] = stuff[int(len(stuff)*(ratio+(1-ratio)*0.5)):]
        else:
            train[c] = stuff
            val[c] = stuff
            test[c] = stuff
    return train, val, test

def get_cores(data, size=-1):
    cores = []
    if size != -1:
        counter = size
        while counter != 0:
            thing = data["things"][random.randint(0, len(data["things"]) - 1)]
            verb = data["transitive_verbs"][random.randint(0, len(data["transitive_verbs"])) - 1]
            agent = data["agents"][random.randint(0, len(data["agents"])) - 1]
            if (agent, verb, thing) not in cores:
                cores.append((agent, verb, thing))
                counter -= 1
    else:
        for thing in data["things"]:
            for agent in data["agents"]:
                for verb in data["transitive_verbs"]:
                    cores.append((agent, verb, thing))
    return cores

def get_word(data,chance, PoS, same=None):
    result = ""
    if random.randint(0,chance):
        result = random.choice(data[PoS])
        if same != None and random.randint(0,chance):
            result = same
    return result

def generate_example(label, prover,data, core1, core2,passive, negate, adjective, adverb):
    determiners = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
    adverb_word = get_word(data,adverb, "adverbs")
    adjective_word1 = get_word(data,adjective, "adjectives1")
    adjective_word2 = get_word(data,adjective, "adjectives2")
    premise = sentence(core1, random.randint(0,passive), random.randint(0,negate), adverb_word, adjective_word1, adjective_word2, determiners)
    determiners = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
    adverb_word = get_word(data, adverb, "adverbs", adverb_word)
    adjective_word1 = get_word(data, adjective, "adjectives1", adjective_word1)
    adjective_word2 = get_word(data, adjective, "adjectives2", adjective_word2)
    hypothesis = sentence(core2, random.randint(0,passive), random.randint(0,negate), adverb_word, adjective_word1, adjective_word2, determiners)
    if random.randint(0,1):
        temp = premise
        premise = hypothesis
        hypothesis = temp
    if label == None:
        label = get_label(prover,premise, hypothesis)
    if False:# label in "contradictsentails":
        print(premise.final)
        print(label)
        print(hypothesis.final)
    return (premise.final, label, hypothesis.final, premise,hypothesis, "")

def temp_fun(x):
    return generate_example(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])

def generate_examples(data, cores, passive, negate, adjective, adverb, distract):
    examples = []
    prover = Prover9()
    prover.config_prover9(r"C:\Program Files (x86)\Prover9-Mace4\bin-win32")
    cores = cores[:1000]
    inputs = [[None, prover, data, core, core,passive, negate, adjective, adverb] for core in cores]
    examples = Parallel(n_jobs=-1,backend="multiprocessing")(map(delayed(temp_fun), inputs))
    random.shuffle(cores)
    for core in cores:
        distract -= 1
        if distract> 0:
            cores = [distraction(core, data), core]
            random.shuffle(cores)
            examples.append(generate_example("permits", prover,data, cores[0], cores[1],passive, negate, adjective, adverb))
    return examples

def distraction(core, data):
    result = [core[0], core[1], core[2]]
    inds = [0,1,2]
    ind = random.choice(inds)
    inds.remove(ind)
    if ind ==0:
        temp = random.choice(data["agents"])
    if ind ==1:
        temp = random.choice(data["transitive_verbs"])
    if ind ==2:
        temp = random.choice(data["things"])
    result[ind] = temp    
    for _ in range(2):
        if random.randint(0,1):
            ind = random.choice(inds)
            inds.remove(ind)
            if ind ==0:
                temp = random.choice(data["agents"])
            if ind ==1:
                temp = random.choice(data["transitive_verbs"])
            if ind ==2:
                temp = random.choice(data["things"])
            result[ind] = temp    
    return tuple(result)


def human_test(data, core, passive, negate):
    examples = []
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                for l in range(0,2):
                    for d1 in ["a", "the","every"]:
                        for d2 in ["a", "the", "every"]:
                            for d3 in ["a", "the", "every"]:
                                for d4 in ["a", "the", "every"]:
                                    passive_premise = i  
                                    negate_premise = j
                                    passive_hypothesis = k
                                    negate_hypothesis = l
                                    premise = sentence(core, passive_premise, negate_premise, data, (d1,d2))
                                    hypothesis = sentence(core, passive_hypothesis, negate_hypothesis, data, (d3,d4))
    #                                label = get_label(prover, premise, hypothesis)
                                    examples.append((premise.final, label, hypothesis.final, premise, hypothesis))
    for e in examples:
        print(e)
    return examples


def generate_boolean_examples(examples, size):
    bool_examples = []
    for i in range(size):
        label = "permits"
        sample = random.sample(examples, 2)
        label = "permits"
        if sample[0][1] == "contradicts" or sample[1][1] == "contradicts":
            label = "contradicts"
        if sample[0][1] == "entails" and sample[1][1] == "entails":
            label = "entails"
        rand = random.randint(0,9)
        while label == "permits" and rand <4:
            sample = random.sample(examples, 2)
            label = "permits"
            if sample[0][1] == "contradicts" or sample[1][1] == "contradicts":
                label = "contradicts"
            if sample[0][1] == "entails" and sample[1][1] == "entails":
                label = "entails"
        order1 = random.randint(0,1)
        order2 = random.randint(0,1)
        premise = sample[order1][0] + " and " + sample[not order1][0]
        hypothesis = sample[order2][2] + " and " + sample[not order2][2]
        bool_examples.append((premise, label, hypothesis, (sample[0][3], sample[1][3]),(sample[0][4], sample[1][4]), "and" ))
    for i in range(size):
        label = "permits"
        sample = random.sample(examples, 2)
        if sample[0][1] == "contradicts" and sample[1][1] == "contradicts":
            label = "contradicts"
        if sample[0][1] == "entails" and sample[1][1] == "entails":
            label = "entails"
        rand = random.randint(0,9)
        while label == "permits" and rand <4:
            sample = random.sample(examples, 2)
            if sample[0][1] == "contradicts" and sample[1][1] == "contradicts":
                label = "contradicts"
            if sample[0][1] == "entails" and sample[1][1] == "entails":
                label = "entails"
        order1 = random.randint(0,1)
        order2 = random.randint(0,1)
        premise = sample[order1][0] + " or " + sample[not order1][0]
        hypothesis = sample[order2][2] + " or " + sample[not order2][2]
        bool_examples.append((premise, label, hypothesis, (sample[0][3], sample[1][3]),(sample[0][4], sample[1][4]), "or" ))
    for i in range(size):
        label = "permits"
        sample = random.sample(examples, 2)
        if sample[0][1] == "entails" and sample[1][1] == "contradicts":
            label = "contradicts"
        if sample[0][1] == "entails" and sample[1][1] == "entails":
            label = "entails"
        rand = random.randint(0,9)
        while label == "permits" and rand <4:
            sample = random.sample(examples, 2)
            if sample[0][1] == "entails" and sample[1][1] == "contradicts":
                label = "contradicts"
            if sample[0][1] == "entails" and sample[1][1] == "entails":
                label = "entails"
        order1 = random.randint(0,1)
        order2 = random.randint(0,1)
        if label != "permits" or sample[0][1] == "contradicts" and sample[1][1] == "entails":
            order1 = 0
            order2 = 0
        premise = "if " + sample[order1][2] + " then " + sample[not order1][0]
        hypothesis = "if " + sample[order2][0] + " then " + sample[not order2][2]
        bool_examples.append((premise, label, hypothesis, (sample[order1][3], sample[not order1][3]),(sample[order2][4], sample[not order2][4]), "ifthen" ))
    for i in range(size):
        label = "permits"
        sample = random.sample(examples, 2)
        if sample[0][1] == "contradicts" and sample[1][1] == "entails":
            label = "entails"
        if sample[0][1] == "contradicts" and sample[1][1] == "contradicts":
            label = "contradicts"
        rand = random.randint(0,9)
        while label == "permits" and rand <4:
            sample = random.sample(examples, 2)
            if sample[0][1] == "contradicts" and sample[1][1] == "entails":
                label = "entails"
            if sample[0][1] == "contradicts" and sample[1][1] == "contradicts":
                label = "contradicts"
        order1 = random.randint(0,1)
        order2 = random.randint(0,1)
        if label != "permits" or sample[0][1] == "entails" and sample[1][1] == "contradicts":
            order2 = 0
        premise = sample[order1][0] + " or " + sample[not order1][0]
        hypothesis = "if " + sample[order2][2] + " then " + sample[not order2][2]
        bool_examples.append((premise, label, hypothesis, (sample[order1][3], sample[not order1][3]),(sample[order2][4], sample[not order2][4]), "ifthenor" ))
    return bool_examples
    

def get_label(prover, premise, hypothesis):
    p = Expression.fromstring(premise.logical_form)
    h = Expression.fromstring(hypothesis.logical_form)
    noth = Expression.fromstring("-" + hypothesis.logical_form)
    if prover.prove(h, [p]):
        return "entails"
    if prover.prove(noth, [p]):
        return "contradicts"
    return "permits"

def save_data(examples, name):
    prem = []
    hyp = []
    label = []
    stuff = []
    for example in examples:
        prem.append(example[0] + "\n")
        label.append(example[1] + "\n")
        hyp.append(example[2] + "\n")
        stuff.append((example[3], example[4], example[5]))
    with open("pi." + "prem." + name, "w") as f:
        f.writelines(prem)
    with open("pi." + "hyp." + name, "w") as f:
        f.writelines(hyp)
    with open("pi." + "label." + name, "w") as f:
        f.writelines(label)
    with open("stuff."+ name+ '.pkl', 'wb') as f:
        pickle.dump(stuff, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    train_data, validation_data, test_data = process_data()
    #print(train_data)
    #print(test_data)
    train_cores = get_cores(train_data,1000)
    validation_cores = get_cores(validation_data,10)
    test_cores = get_cores(test_data,10)
    #human_test(test_data, test_cores[0],1,1)
    if True:
        train_examples = generate_examples(train_data, train_cores,1,1,0,1, len(train_cores)/5)
        validation_examples = generate_examples(validation_data, validation_cores,1,1,0,1,len(validation_cores)/5)
        test_examples = generate_examples(test_data, test_cores,1,1,0,1, len(test_cores)/5)
        print(train_examples)
        print("next")
        print(len(validation_examples))
        print("next")
        print(len(test_examples))
        count = 0
        count1 = 0
        for example in train_examples:
            if example[1] == "permits":
                count +=1
            if example[1] == "entails":
                count1 +=1
        print(count/len(train_examples))
        print(count1/len(train_examples))
        print(1 - (count + count1)/len(train_examples))
        train_examples += generate_boolean_examples(train_examples, len(train_examples))
        validation_examples += generate_boolean_examples(validation_examples, len(validation_examples))
        test_examples += generate_boolean_examples(test_examples, len(test_examples))
        print(len(train_examples))
        print("next")
        print(len(validation_examples))
        print("next")
        print(len(test_examples))
        count = 0
        count1 = 0
        for example in train_examples:
            if example[1] == "permits":
                count +=1
            if example[1] == "entails":
                count1 +=1
        print(count/len(train_examples))
        print(count1/len(train_examples))
        print(1 - (count + count1)/len(train_examples))
        random.shuffle(train_examples)
        save_data(train_examples[50000:], "train")
        save_data(validation_examples, "val")
        save_data(test_examples, "test")
        save_data(train_examples[0:50000], "jointtest")
