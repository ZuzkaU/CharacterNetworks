import spacy
from spacy.tokens import Doc, Token, Span, DocBin
import logging
from tqdm import tqdm

import annotation.coref
from annotation.coref import CorefModel
import annotation.entity_modifier
from annotation.entity_modifier import EntityModifier
import annotation.quote_parser
from annotation.quote_parser import QuoteParser

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter

class Annotator:

    def __init__(self):
        logging.info("Loading language...")
        nlp = spacy.load("en_core_web_trf")
        
        logging.info("Preparing pipe...")
        nlp.add_pipe("entity_modifier")
        nlp.add_pipe("quote_parser")
        
        self.nlp = nlp
        
        logging.info("Loading coreference model...")
        self.coref = CorefModel()
        
        Annotator.setExtensions()
        
        with open("vocab/character-verb-predicates.tsv") as f:
            lines = f.read().splitlines()
            split_lines = [line.split('\t') for line in lines]
            self.character_verb_predicates = {}
            for (verb, dep) in split_lines:
                if verb in self.character_verb_predicates:
                    self.character_verb_predicates[verb].append(dep)
                else:
                    self.character_verb_predicates[verb] = [dep]
        
        with open("vocab/stop-list.txt") as f:
            lines = f.read().splitlines()
            self.stoplist = lines

    def setExtensions():
        Span.set_extension("nameless_name", default=None)

    def getSlicesForCoref(self, docs, max_len):
        """
        Groups docs to slices with cumulative length of at most max_len (except when
        docs are long: at least one new doc and max_len/2 of previous docs are in
        one slice). A doc can be in multiple slices, it is done as a sliding window.
        
        Returns:
            [(start, end)]: List of tuples of docs indexes, these docs slices may
                            be merged for coreference resolution on a longer text.
        """
        slices = []
        first_doc, last_doc = 0, 0
        current_len = 0
        
        while last_doc < len(docs):
            # add one new doc
            first_doc = last_doc
            last_doc += 1
            current_len = len(docs[first_doc])
            
            # add old docs up to max_len/2 (-> sliding window)
            while first_doc > 0 and current_len + len(docs[first_doc-1]) < max_len/2:
                first_doc -= 1
                current_len += len(docs[first_doc])
            
            # add new docs up to max_len
            while last_doc < len(docs) and current_len + len(docs[last_doc]) < max_len:
                current_len += len(docs[last_doc])
                last_doc += 1
            
            slices.append((first_doc, last_doc))
        
        return slices


    def annotate(self, paragraphs):
        logging.info("Tokenizing the document...")
        logging.info("This might take a few minutes.")
        
        docs = list(self.nlp.pipe(paragraphs))
        
        coref_slices = self.getSlicesForCoref(docs, self.coref.MAX_LEN)
        for (i, j) in tqdm(coref_slices, desc="Resolving coreference", unit="slice"):
            self.coref(docs[i:j])
        
        
        nameless_characters = {}
        lemmatizer = WordNetLemmatizer()
        for doc_i, doc in enumerate(docs):
            for token in doc:
                if token.pos_ == "VERB" and token.lemma_ in self.character_verb_predicates:
                    for child in token.children:
                        if child.pos_ == "NOUN" or child.pos_ == "PROPN" and child.dep_ in self.character_verb_predicates[token.lemma_]:
                            if not (child.ent_iob_ == "B" or child.ent_iob_ == "I"):
                                singular = lemmatizer.lemmatize(child.lower_)
                                if not singular == child.lower_:
                                    continue
                                try:
                                    synset = wn.synset(singular + '.n.01')
                                except:
                                    continue
                                is_person = wn.synset('organism.n.01') in synset.lowest_common_hypernyms(wn.synset('organism.n.01'))
                                if is_person and not singular in self.stoplist and not child._.is_relation:
                                    if child.text in nameless_characters:
                                        nameless_characters[child.text][0].append(((doc_i, child.i)))
                                    else:
                                        nameless_characters[child.text] = ([(doc_i, child.i)], [])
                                    chunk = None
                                    for ch in doc.noun_chunks:
                                        if ch.root == child:
                                            nameless_characters[child.text][1].append(ch)
                                    
        for nameless_id, (name, (occur_list, chunks)) in enumerate(nameless_characters.items()):
            chunk = Counter([ch.text for ch in chunks]).most_common()
            if chunk and chunk[0][1] >= 3:
                for (doc_id, tok_id) in occur_list:
                    span = Span(docs[doc_id], tok_id, tok_id+1, "NAMELESS_CHAR")
                    docs[doc_id].set_ents(list(docs[doc_id].ents) + [span])
                    span._.nameless_name = chunk[0][0]
            
        
        for doc in docs:
            for token in doc:
                if token.lower_ in ['i', 'me', 'my'] and not token._.is_direct_speech:
                    if not (token.ent_iob_ == "B" or token.ent_iob_ == "I"):
                        span = Span(doc, token.i, token.i+1, "NARRATOR")
                        doc.set_ents(list(doc.ents) + [span])
        
        return docs


class FalseAnnotator(Annotator):
    def __init__(self):
        logging.info("Loading language...")
        self.nlp = spacy.load("en_core_web_trf")
        
        logging.info("Setting extensions...")
        CorefModel.setExtensions()
        EntityModifier.setExtensions()
        QuoteParser.setExtensions()
        Annotator.setExtensions()
    
        
    def annotate(self, docbin_file):
        logging.info("Loading docs from file '{}'...".format(docbin_file))
        doc_bin = DocBin().from_disk(docbin_file)
        docs = list(doc_bin.get_docs(self.nlp.vocab))
        return docs
            
