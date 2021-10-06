#! /usr/bin/env python3

import os
import sys
root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(os.path.join(root_dir, 'long-doc-coref/src'))
from inference.inference import Inference

import logging
import spacy
from spacy.tokens import Doc, Span, Token


class CorefModel:
    def __init__(self):
        
        self.model = Inference(os.path.join(root_dir, 'models', 'coref.pth'))
        self.MAX_LEN = 512
        self.current_cluster_id = 0
        
        CorefModel.setExtensions()
        return
    
    def setExtensions():
        Doc.set_extension("cluster_ids", default=None)
        Doc.set_extension("coref_ents", default=[])
        
        Token.set_extension("clusters", default=[])
        
        def isInSpans(span, spans):
            for other_span in spans:
                if span.start == other_span.start and span.end == other_span.end:
                    return True
            return False
        Span.set_extension("isIn", method=isInSpans)
    
    def __call__(self, docs):
        """
        Performs coreference resolution on the given list of docs.
        Appends to the following attributes:
            span._.clusters
            doc._.coref_ents
            doc._.cluster_ids at docs
        
        Args:
            docs: List of spacy doc
        """
        output = self.model.perform_coreference(' '.join([doc.text for doc in docs]))
        
        for doc in docs:
            if not doc._.cluster_ids:
                doc._.cluster_ids = (self.current_cluster_id, self.current_cluster_id)
        
        spacy_mapping = self.getSpacyIndexes(docs, output["tokenized_doc"])
        
        for cluster in output["clusters"]:
            if len(cluster) == 1:
                continue
            for ((start, end), text) in cluster:
                span, doc = self.getDocSpan(docs, start, end, spacy_mapping, text)
                span.root._.clusters.append(self.current_cluster_id)
                doc._.coref_ents.append((span.start, span.end, span.text, self.current_cluster_id))
            self.current_cluster_id += 1
        
        
        for doc in docs:
            (start, end) = doc._.cluster_ids
            doc._.cluster_ids = (start, self.current_cluster_id)
        
        
        return
    
    
    def getSpacyIndexes(self, docs, tokenized_doc):
        """
        Aligns token indexes for different tokenizers of spacy and coreference
        resolution tool.
        
        Returns:
            [int]: 
        """
        doc_id, token_id = 0, 0
        i = 0
        
        tokens = self.getTokensFromSubtokens(tokenized_doc)
        spacy_mapping = [None] * (len(tokens)) # list of tuples (doc_id, token_id), index = id of coref token
        
        def increase(doc_id, token_id):
            token_id += 1
            if token_id >= len(docs[doc_id]):
                token_id = 0
                doc_id += 1
            return (doc_id, token_id)
        
        while i < len(tokens):
            token = tokens[i]
            spacy_token_text = docs[doc_id][token_id].text
            
            if token == spacy_token_text:
                spacy_mapping[i] = (doc_id, token_id)
            else:
                spacy_text, bert_text = spacy_token_text, token
                while not spacy_text == bert_text:
                    
                    if spacy_text.startswith(bert_text):
                        spacy_mapping[i] = spacy_mapping[i] if spacy_mapping[i] else (doc_id, token_id)
                        i += 1
                        if i >= len(tokens):
                            break
                        token = tokens[i]
                        bert_text += token
                    elif bert_text.startswith(spacy_text):
                        spacy_mapping[i] = spacy_mapping[i] if spacy_mapping[i] else (doc_id, token_id)
                        doc_id, token_id = increase(doc_id, token_id)
                        spacy_token_text = docs[doc_id][token_id].text
                        spacy_text += spacy_token_text
                    elif spacy_text.isspace():
                        spacy_text = ""
                    else:
                        logging.warning("ERROR: Tokens not aligned: %s %s", spacy_text, bert_text)
                        break
                if i < len(tokens):
                    spacy_mapping[i] = spacy_mapping[i] if spacy_mapping[i] else (doc_id, token_id)
            
            doc_id, token_id = increase(doc_id, token_id)
            i += 1
        return spacy_mapping
    
    
    def getTokensFromSubtokens(self, tokenized_doc):
        subtoken_map = tokenized_doc["subtoken_map"]
        subtoken_id = 0
        token_id = 0
        tokens = []
        current_token = []
        
        def concatenateToken(parts):
            token = parts[0]
            if len(parts) > 1:
                for part in parts[1:]:
                    assert part[:2] == "##"
                    token += part[2:]
            return token
        
        for sentence in tokenized_doc["sentences"]:
            for subtoken in sentence:
                if subtoken_map[subtoken_id] == token_id:
                    current_token.append(subtoken)
                else:
                    tokens.append(concatenateToken(current_token))
                    token_id += 1
                    current_token = [subtoken]
                subtoken_id += 1
        
        if current_token:
            tokens.append(concatenateToken(current_token))
        
        return tokens
    
    
    def getDocSpan(self, docs, start, end, mapping, text):
        doc1, token1 = mapping[start]
        doc2, token2 = mapping[end]
        
        if not doc1 == doc2:
            logging.debug("Docs not equal: %s, %s\n\t%s\n\t%s", doc1, doc2, docs[doc1][token1:], docs[doc2][:token2+1])
        
        doc = docs[doc1]
        span = doc[token1:token2+1]
        
        if not text == span.text:
            logging.debug("Spans not equal: \n\t%s\n\t%s", text, span.text)
        
        return span, doc

