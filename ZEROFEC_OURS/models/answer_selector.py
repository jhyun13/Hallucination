from typing import  Dict
import spacy
import stanza
import pandas as pd
import time


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
        This function processes each atomic text in the data['atomic_text'] list
        and adds candidate answers for each atomic text to the DataFrame.
        '''
        print("[Answer Selector] Selecting answers for atomic texts ...\n")

        # Initialize lists to store new columns
        candidate_answers_list = []
        latency_list = []

        for atomic_text in data['atomic_text']:
            start_time = time.time()
            
            # Process each atomic text
            doc = self.nlp(atomic_text)
            stanza_doc = self.stanza_nlp(atomic_text)

            ents = [ent.text for sent in doc.sents for ent in sent.noun_chunks] 
            ents += [ent.text for sent in doc.sents for ent in sent.ents]
            ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'NP')]
            ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'VP')]
            ents += [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos in ['VERB','ADV','ADJ','NOUN']]

            # Negation
            negations = [word for word in ['not', 'never'] if word in atomic_text]

            # Look for middle part: relation/verb
            middle = []
            start_match = ''
            end_match = ''
            for ent in ents:
                # Look for longest match string
                if atomic_text.startswith(ent) and len(ent) > len(start_match):
                    start_match = ent
                if atomic_text.endswith(ent + '.') and len(ent) > len(end_match):
                    end_match = ent

            if len(start_match) > 0 and len(end_match) > 0:
                middle.append(atomic_text[len(start_match):-len(end_match)-1].strip())

            # Combine all candidate answers
            candidate_answers = list(set(ents + negations + middle))

            # Append to the new column list
            candidate_answers_list.append(candidate_answers)
            
            end_time = time.time()
            
            latency = end_time - start_time
            latency_list.append(latency)

        # Add new columns to the original DataFrame
        data['candidate_answers'] = candidate_answers_list
        data['candidate_answer_latency'] = latency_list

        return data