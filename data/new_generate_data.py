import random
import os
import copy
import pickle
from nltk.sem.logic import *
from nltk.inference import *
from nltk import Prover9
from joblib import Parallel, delayed

class sentence:
    #this class stores the logical representation of a sentence and the natural language representation of a sentence
    def __init__(self, core, passive, negate, adverb, adjective1, adjective2, determiners, adverb_position):
        self.core = core
        self.passive = passive
        self.negate = negate
        self.determiners = determiners
        self.negation = ["", ""]
        self.adverb = adverb 
        self.adjective1 = adjective1
        self.adjective2 = adjective2
        self.adverb_position = adverb_position
        self.negate_adverb = ""
        self.verb_adverb = ""
        self.sentence_adverb = ""
        if self.adverb_position == "before negation":
            assert self.adverb != ""
            self.negate_adverb = self.adverb
            self.sentence_adverb = ""
            self.verb_adverb = ""
        if self.adverb_position == "before verb":
            assert self.adverb != ""
            self.verb_adverb = self.adverb
        if self.adverb_position == "end":
            self.sentence_adverb = self.adverb
            self.adverb_position = "end"
        self.non_passive_verb_index = 0
        if negate:
            self.negation = ["not", "does not"] #negation for passive and active
            self.non_passive_verb_index = 2#controls the verb form
        if self.passive:
            temp = self.adjective1
            self.adjective1 = self.adjective2#This ensures that identical adjectives appearing in
            self.adjective2 = temp           # both the premise and hypothesis apply to the same noun
            self.final = self.construct_string([self.determiners[0],self.adjective1,self.core[2],"is",self.negate_adverb,self.negation[0],self.verb_adverb,self.core[1][1],"by",self.determiners[1],self.adjective2,self.core[0],self.sentence_adverb])
        else:
            self.final = self.construct_string([self.determiners[0],self.adjective1,self.core[0],self.negate_adverb,self.negation[1],self.verb_adverb,self.core[1][self.non_passive_verb_index],self.determiners[1],self.adjective2,self.core[2],self.sentence_adverb])
        self.change_an() #replaces a with an if needed
        self.construct_logical_form()

    def change_an(self):
        #replace a with an where necessary
        if self.final[0] == "a" and self.final[2] in "aeiou":
            self.final = "an" + self.final[1:]
        for i in range(len(self.final) - 2):
            if self.final[i:i+3] == " a " and self.final[i+3] in "aeiou":
                self.final = self.final[:i+2] + "n" + self.final[i+2:]
                

    def construct_string(self,lst):
        #turn a list of words into a single sentence string
        result = ""
        for word in lst:
            if word != "":
                result += word + " "
        return result[:-1]

    def construct_logical_form(self):
        #construct a first order logic representation
        logical_form = ""
        append_formula = []
        working_core = list(copy.copy(self.core))
        verb = working_core[1][0]
        constant1 = "(constant1)"
        constant2 = "(constant2)"
        if self.passive: #ensures that subjects and objects have the same constants
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
        logical_form = "(" + logical_form + ")"
        for form in append_formula:
            logical_form = logical_form + "&" + form
        self.logical_form = logical_form 


def process_data(train_ratio):
    #split the different parts of speech into train, validation, and test
    #determiners are not split
    train = dict()
    val = dict()
    test = dict()
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
            train[c] = stuff[:int(len(stuff)*train_ratio)]
            val[c] = stuff[int(len(stuff)*train_ratio):int(len(stuff)*(train_ratio+(1-train_ratio)*0.5))]
            test[c] = stuff[int(len(stuff)*(train_ratio+(1-train_ratio)*0.5)):]
        else:
            train[c] = stuff
            val[c] = stuff
            test[c] = stuff
    return train, val, test

def get_cores(data, size=-1):
    #constructs a list of length size where each element
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

def get_word(data,chance, PoS, same=""):
    #if chance is 0 the empty string is returned
    #if chance is 1 and same is None then a random word of part of speech PoS is returned
    #otherwise there is a 0.5 chance that same is returned and an 0.5 chance that a random word of part of speech PoS is returned
    result = ""
    if random.randint(1,10) <= chance:
        result = random.choice(data[PoS])
    if same != "":
        temp = random.randint(1,3)
        if temp == 1:
            return same
        if temp == 2:
            return random.choice(data[PoS])
        if temp == 3:
            return ""
    return result

def get_label(prover, premise, hypothesis):
    #returns a label that is determined from using Prover9 on the first order logic representations
    p = Expression.fromstring(premise.logical_form)
    h = Expression.fromstring(hypothesis.logical_form)
    noth = Expression.fromstring("-" + hypothesis.logical_form)
    if prover.prove(h, [p]):
        return "entails"
    if prover.prove(noth, [p]):
        return "contradicts"
    return "permits"

def generate_random_example(inputs):
    #returns a tuple of premise string, label, hypothesis string, premise object, hypothesis object
    label, prover,data, core1, core2,passive, negate, adjective, adverb = inputs
    determiners = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
    adverb_word = get_word(data,adverb, "adverbs")
    adjective_word1 = get_word(data,adjective*6, "adjectives1")
    adjective_word2 = get_word(data,adjective*6, "adjectives2")
    passive_value = random.randint(0,passive)
    negation_value = random.randint(0,negate)
    adverb_location = "" 
    if adverb_word != "":
        adverb_location = ["end", "before verb"][random.randint(0,1)]
        if negation_value:
            adverb_location = ["end", "before verb", "before negation"][random.randint(0,2)]
    premise = sentence(core1, passive_value, negation_value, adverb_word, adjective_word1, adjective_word2, determiners, adverb_location)
    determiners = tuple([data["determiners"][random.randint(0, len(data["determiners"]) - 1)] for i in range(2)])
    adverb_word = get_word(data, adverb, "adverbs", adverb_word)
    adjective_word1 = get_word(data, adjective*5, "adjectives1", adjective_word1)
    adjective_word2 = get_word(data, adjective*5, "adjectives2", adjective_word2)
    passive_value = random.randint(0,passive)
    negation_value = random.randint(0,negate)
    adverb_location = "" 
    if adverb_word != "":
        adverb_location = ["end", "before verb"][random.randint(0,1)]
        if negation_value:
            adverb_location = ["end", "before verb", "before negation"][random.randint(0,2)]
    hypothesis = sentence(core2, passive_value, negation_value, adverb_word, adjective_word1, adjective_word2, determiners, adverb_location)
    if label == None:
        label = get_label(prover,premise, hypothesis)
    print(premise.final)
    return (premise.final, label, hypothesis.final, {"premise":[premise],"hypothesis":[hypothesis]})

def build_file(data):
    prover = Prover9()
    prover.config_prover9(r"C:\Program Files (x86)\Prover9-Mace4\bin-win32")
    core = ["man", ["eats", "eaten", "eat"], "rock"]
    dets = ["a", "the", "every"]
    count = 0
    for pd1_index in range(3):
        pd1 = dets[pd1_index]
        for pd2_index in range(3):
            pd2 = dets[pd2_index]
            for hd1_index in range(3):
                hd1 = dets[hd1_index]
                for hd2_index in range(3):
                    hd2 = dets[hd2_index]
                    for ppassive_value in range(2):
                        for hpassive_value in range(2):
                            for padj1 in range(2):
                                if padj1 == 0:
                                    padj1_word = ""
                                else:
                                    padj1_word = random.choice(data["adjectives1"])
                                for hadj1 in range(2):
                                    if hadj1 == 0:
                                        hadj1_word = ""
                                    else:
                                        hadj1_word = ""
                                        while hadj1_word == padj1_word or hadj1_word == "":
                                            hadj1_word = random.choice(data["adjectives1"])
                                    for same_adj1 in range(2):
                                        if same_adj1 == 1 and (padj1 == 0 or hadj1 == 0):
                                            continue
                                        elif same_adj1 ==1:
                                            hadj1_word = padj1_word
                                        for padj2 in range(2):
                                            if padj2 == 0:
                                                padj2_word = ""
                                            else:
                                                padj2_word = random.choice(data["adjectives2"])
                                            for hadj2 in range(2):
                                                if hadj2 == 0:
                                                    hadj2_word = ""
                                                else:
                                                    hadj2_word = ""
                                                    while hadj2_word == padj2_word or hadj2_word == "":
                                                        hadj2_word = random.choice(data["adjectives2"])
                                                for same_adj2 in range(2):
                                                    if same_adj2 == 1 and (padj2 == 0 or hadj2 == 0):
                                                        continue
                                                    elif same_adj2 ==1:
                                                        hadj2_word = padj2_word
                                                    for padv in range(2):
                                                        if padv == 0:
                                                            padv_word = ""
                                                        else:
                                                            padv_word = random.choice(data["adverbs"])
                                                        for hadv in range(2):
                                                            if hadv == 0:
                                                                hadv_word = ""
                                                            else:
                                                                hadv_word = ""
                                                                while hadv_word == padv_word or hadv_word == "":
                                                                    hadv_word = random.choice(data["adverbs"])
                                                            for same_adv in range(2):
                                                                if same_adv == 1 and (padv == 0 or hadv == 0):
                                                                    continue
                                                                elif same_adv ==1:
                                                                    hadv_word = padv_word
                                                                for pnegation_value in range(2):
                                                                    for hnegation_value in range(2):
                                                                        for padv_location_index in range(4):
                                                                            padv_location = ["","before verb", "before negation", "end"][padv_location_index]
                                                                            if (padv_location_index == 2 and pnegation_value == 0) or (padv_location_index != 0 and padv ==0):
                                                                                continue
                                                                            for hadv_location_index in range(4):
                                                                                hadv_location = ["","before verb", "before negation", "end"][hadv_location_index]
                                                                                if (hadv_location_index == 2 and hnegation_value == 0) or (hadv_location_index != 0 and hadv ==0):
                                                                                    continue
                                                                                count += 1
                                                                                #hypothesis = sentence(core, hpassive_value, hnegation_value, hadv_word, hadj1_word, hadj2_word, [hd1,hd2], hadverb_location)
                                                                                #premise = sentence(core, ppassive_value, pnegation_value, padv_word, padj1_word, padj2_word, [pd1,pd2], padverb_location)
                                                                                #label = get_label(prover,premise, hypothesis)
                                                                                #result[(hpassive_value, hnegation_value, hadv, hadj1, hadj2, hd1_index,hd2_index, hadverb_location_index,ppassive_value, pnegation_value, padv, padj1, padj2, pd1_index,pd2_index, padverb_location_index, same_adj1, same_adj2, same_adv)]
    print(count)


def generate_examples(data, cores, passive, negate, adjective, adverb, distract):
    #returns a list of examples the same length as cores
    examples = []
    prover = Prover9()
    prover.config_prover9(r"C:\Program Files (x86)\Prover9-Mace4\bin-win32")
    cores = cores
    inputs = [[None, prover, data, core, core,passive, negate, adjective, adverb] for core in cores] #generate examples in parallel
    examples = Parallel(n_jobs=-1,backend="multiprocessing")(map(delayed(generate_random_example), inputs))
    random.shuffle(cores)
    for core in cores:
        distract -= 1
        if distract> 0:
            cores = [distraction(core, data), core]
            random.shuffle(cores)
            examples.append(generate_random_example(["permits", prover,data, cores[0], cores[1],passive, negate, adjective, adverb]))
    return examples

def distraction(core, data):
    #returns a tuple that has at least one element different from core
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

def generate_compound(examples, first_predicate, second_predicate, sample, sample_index, conjunction, logic_operator):
    order = random.randint(0,1)
    if order:
        temp = first_predicate
        first_predicate = second_predicate
        second_predicate = temp
    sentence = sample[order][sample_index] + " " + conjunction + " " + sample[not order][sample_index]
    if conjunction == "then":
        sentence = "if " + sentence 
    logic_form = "(" + first_predicate+"(constant)"+logic_operator + second_predicate+"(constant)" + ")"
    return sentence, logic_form, order

def generate_boolean_examples(examples, size):
    prover = Prover9()
    prover.config_prover9(r"C:\Program Files (x86)\Prover9-Mace4\bin-win32")
    bool_examples = []
    conjunctions = ["or", "and", "then"]
    logic_operators = ["|", "&", "->"]
    for i in range(size):
        sample = random.sample(examples, 2)
        index = random.randint(0,2)
        prem_conjunction = conjunctions[index]
        logic_operator = logic_operators[index]
        premise, first_assumption, prem_order  = generate_compound(examples, "A", "B", sample, 0, prem_conjunction, logic_operator)
        index = random.randint(0,2)
        hyp_conjunction = conjunctions[index]
        logic_operator = logic_operators[index]
        hypothesis, conclusion, hyp_order = generate_compound(examples, "C", "D", sample, 2, hyp_conjunction, logic_operator)
        assumptions = [Expression.fromstring(first_assumption)]
        if sample[0][1] == "entails":
            assumptions.append(Expression.fromstring("A(constant)->C(constant)"))
        if sample[0][1] == "contradicts":
            assumptions.append(Expression.fromstring("-A(constant)|-C(constant)"))
        if sample[1][1] == "entails":
            assumptions.append(Expression.fromstring("B(constant)->D(constant)"))
        if sample[1][1] == "contradicts":
            assumptions.append(Expression.fromstring("-B(constant)|-D(constant)"))
        label = None 
        if prover.prove(Expression.fromstring(conclusion), assumptions):
            label = "entails"
        if prover.prove(Expression.fromstring("-"+conclusion), assumptions):
            label = "contradicts"
        else:
            label = "permits"
        bool_examples.append((premise, label, hypothesis, {"premise":[sample[prem_order][3]["premise"][0],sample[not prem_order][3]["premise"][0]],"hypothesis":[sample[hyp_order][3]["hypothesis"][0],sample[not hyp_order][3]["hypothesis"][0]], "prem_conjunction":prem_conjunction, "hyp_conjunction":hyp_conjunction}))
    return bool_examples
    


def save_data(examples, name):
    prem = []
    hyp = []
    label = []
    stuff = []
    for example in examples:
        prem.append(example[0] + "\n")
        label.append(example[1] + "\n")
        hyp.append(example[2] + "\n")
        stuff.append((example[3]))
    with open("pi." + "prem." + name, "w") as f:
        f.writelines(prem)
    with open("pi." + "hyp." + name, "w") as f:
        f.writelines(hyp)
    with open("pi." + "label." + name, "w") as f:
        f.writelines(label)
    with open("stuff."+ name+ '.pkl', 'wb') as f:
        pickle.dump(stuff, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    train_data, validation_data, test_data = process_data(0.6)
    build_file(train_data)
    #print(train_data)
    #print(test_data)
    train_cores = get_cores(train_data,100)
    validation_cores = get_cores(validation_data,10)
    test_cores = get_cores(test_data,10)
    print(len(train_cores))
    #human_test(test_data, test_cores[0],1,1)
    if True:
        train_examples = generate_examples(train_data, train_cores,1,1,1,1, len(train_cores)/5)
        validation_examples = generate_examples(validation_data, validation_cores,1,1,1,1,len(validation_cores)/5)
        test_examples = generate_examples(test_data, test_cores,1,1,1,1, len(test_cores)/5)
        print(train_examples)
        print("next1")
        print(len(validation_examples))
        print("next2")
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
        print(train_examples)
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
