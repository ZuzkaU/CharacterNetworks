import spacy
from spacy.tokens import Doc, Span

import quote_attribution.quote_mention
import quote_attribution.mention_speaker
import evaluation.quotes_evaluation

import logging

class Sieve:
    pass

class QuoteAttributor:
    def __init__(self, docs, characters):
        self.docs = docs
        self.characters = characters
        
        self.name_dict = {}
        self.gender_dict = {}
        for char_id, (var_list, gender) in characters.items():
            for (name, count) in var_list:
                self.name_dict[name] = char_id
            self.gender_dict[char_id] = gender
        character_set = set(self.name_dict.keys())
        
        
        QuoteAttributor.setExtensions()
        
        self.qo_me = quote_attribution.quote_mention.QuoteMention(character_set, docs)
        self.me_sp = quote_attribution.mention_speaker.MentionSpeaker(self.name_dict, self.gender_dict, docs)
    
    def setExtensions():
        quote_attribution.quote_mention.QuoteMention.setExtensions()
        quote_attribution.mention_speaker.MentionSpeaker.setExtensions()
        
    
    def extractSpeakers(self):
        self.qo_me.solveMentions()
        self.me_sp.solveSpeakers()
        
        logging.info('Speaker attribution done.')
        return self.docs
    
class FalseQuoteAttributor:
    def __init__(self, docs, characters, quoteli3_file):
        QuoteAttributor.__init__(self, docs, characters)
    
        self.evaluator = evaluation.quotes_evaluation.QuotesEvaluatorQuoteLi3(docs, quoteli3_file, characters)
        return
        
    def extractSpeakers(self):
        QuoteAttributor.extractSpeakers(self)
        self.docs = self.evaluator.addGoldSpeakers()
        
        return self.docs

