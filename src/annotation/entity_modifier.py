#! /usr/bin/env python3

import os
import json
import spacy
from spacy import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
import logging


@Language.factory("entity_modifier")
def createEntityModifier(nlp, name):
    return EntityModifier(nlp.vocab)


class EntityModifier:
    def __init__(self, vocab):
        EntityModifier.setExtensions()
        
        vocab_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'vocab')
        
        with open(os.path.join(vocab_dir, 'honorific.json')) as honorific_file:
            honorific = json.load(honorific_file)
        
        with open(os.path.join(vocab_dir, 'relations.json')) as relations_file:
            relations = json.load(relations_file)
        
        patterns_honorific = [[{"TEXT" : honorific}] for honorific in honorific["woman"] + honorific["man"] + honorific["other"]]
        patterns_relation = [[{"TEXT" : rel}] for rel in relations["woman"] + relations["man"] + relations["other"]]
        patterns_woman = [[{"TEXT" : woman}] for woman in honorific["woman"] + relations["woman"]]
        patterns_man = [[{"TEXT" : man}] for man in honorific["man"] + relations["man"]]
        
        self.matcher = Matcher(vocab)
        self.matcher.add("HONORIFIC", patterns_honorific)
        self.matcher.add("RELATION", patterns_relation)
        self.matcher.add("WOMAN", patterns_woman)
        self.matcher.add("MAN", patterns_man)
        return
    
    def setExtensions():
        Token.set_extension("is_honorific", default=False)
        Token.set_extension("is_relation", default=False)
        Token.set_extension("is_woman", default=False)
        Token.set_extension("is_man", default=False)
    
    def __call__(self, doc):
        matches = self.matcher(doc)
        doc = addExtensions(doc, matches)
        doc = modifyEntities(doc)
        return doc


def addExtensions(doc, matches):
    for match_id, start, end in matches:
        if doc.vocab.strings[match_id] == "HONORIFIC":
            doc[start]._.is_honorific = True
        elif doc.vocab.strings[match_id] == "RELATION":
            doc[start]._.is_relation = True
        elif doc.vocab.strings[match_id] == "WOMAN":
            doc[start]._.is_woman = True
        elif doc.vocab.strings[match_id] == "MAN":
            doc[start]._.is_man = True
        else:
            logging.warning("Match ID not found!")
    return doc


def modifyEntities(doc):
    """
    Removes entities in the doc not containing persons, 
    modifies person entities to contain also honorifics
    """
    modified_entities = []
    for ent in doc.ents:
        if(ent.label_ == "PERSON"):
            if ent.start > 0 and doc[ent.start - 1]._.is_honorific:
                modified_entities.append(Span(doc, ent.start - 1, ent.end, "PERSON"))
            else:
                modified_entities.append(ent)
    doc.set_ents(modified_entities)
    return doc
