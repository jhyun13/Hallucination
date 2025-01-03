from typing import  Dict
import spacy
import stanza
import pandas as pd


def get_phrases(tree, label):
    if tree.is_leaf():
        return []
    results = [] 
    for child in tree.children:
        results += get_phrases(child, label)
    
    
    if tree.label == label:
        return [' '.join(tree.leaf_labels())] + results
    else:
        return results

class AnswerSelector:
    def __init__(self, args):
        
        # args used during generation
        self.args = args
        if self.args.use_scispacy:
            self.nlp = spacy.load('en_core_sci_md')
            
        else:
            self.nlp = spacy.load('en_core_web_lg')
        self.stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        
        print("[Answer Selector] Initialized.\n")

    

    def select_answers(self, data: pd.DataFrame):
        '''
        This function delete time-related information and store it in `time_removed_tweet`.
        '''
        print("[Answer Selector] Selecting answer for atomic text  ...\n")
        
        doc = self.nlp(data['atomic_text'])
        stanza_doc = self.stanza_nlp(data['atomic_text'])
        
        ents = [ent.text for sent in doc.sents for ent in sent.noun_chunks] 
        ents += [ent.text  for sent in doc.sents for ent in sent.ents]
        ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'NP')]
        ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'VP')]
        ents += [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos in ['VERB','ADV','ADJ','NOUN']]
        


        # negation
        negations = [word for word in ['not','never'] if word in data['atomic_text']]

        # look for middle part: relation/ verb
        middle = []
        start_match = ''
        end_match = ''
        for ent in ents:
            # look for longest match string
            if data['atomic_text'].startswith(ent) and len(ent) > len(start_match):
                start_match = ent
            if data['atomic_text'].endswith(ent+'.') and len(ent) > len(end_match):
                end_match = ent
        
        
        if len(start_match) > 0 and len(end_match) > 0:
            
            middle.append(data['atomic_text'][len(start_match):-len(end_match)-1].strip())
            
        data['candidate_answers'] = list(set(ents + negations + middle))

        return data
