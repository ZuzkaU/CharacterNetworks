#! /usr/bin/env python3
import random
import logging
from tqdm import tqdm
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Token, Span
import os
import json

import character_extraction.name_parser as name_parser
import character_extraction.name_unification_model as name_unification_model
import character_extraction.name_unification_graph as name_unification_graph


class CharacterExtractor:
    def __init__(self, docs):
        """
        Args:
            docs: list of spacy Doc covering the whole novel
        
        Graph edge types:
            coref_connected     : number of occurences in the same cluster
            coref_unconnected   : number of occurences in different clusters
            gender_same         : 1 if the inferred gender is the same
            gender_different    : 1 if the inferred gender is different
            conjunction         : number of connections of the names by conjunction
        Additional only for proper names (not unnamed characters):
            honorific_differ    : 1 if both names contain honorific and they differ
            name_subset         : 1 if one name is a subset of the other (e.g. omitted honorific)
            first_name_variant  : 1 if both have first names and are variants from a gazeteer
            name_part_differ    : 1 if the names cannot be merged (e.g. contain different first names)
        """
        CharacterExtractor.setExtensions()
        self.docs = docs
        self.clusters = self.reconstructClusters()
        
        self.gendered_words = {}
        vocab_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'vocab')
        with open(os.path.join(vocab_dir, 'gendered_words.json')) as genders_file:
            gendered_words_list = json.load(genders_file)
        for item in gendered_words_list:
            self.gendered_words[item['word']] = item['gender'].upper()
        
    
    
    def extractCharacters(self, model_path="name_unification.model", edge_maxprob=0.9, character_remove_limit=3):
        """
        Extracts and merges characters.
        
        When finished, character clusters are available in self.characters
        """
        logging.info("Begin extracting characters...")
        
        G = name_unification_graph.CharacterUnificationGraph(self.docs, self.clusters).createGraph()
        
        edge_prediction = self.predictProba(G, model_path)
        final_graph = self.initFinalGraph(G, edge_prediction)
        final_graph = self.removeNodes(final_graph, G, edge_prediction, edge_maxprob, character_remove_limit)
        final_graph, G = self.renameNodes(final_graph, G, edge_prediction)
        self.final_graph = final_graph
        self.G = G
        
        self.characters = self.makeCharacters(final_graph, G)
        
        self.markCharactersInCoref(self.characters)
        
        logging.info("Characters extracted.")
        
        return self.characters
    
    
    def reconstructClusters(self):
        mindoc, maxdoc = 0, 0
        clusters = []
        for cluster_id in range(self.docs[-1]._.cluster_ids[1]):
            while maxdoc < len(self.docs) and self.docs[maxdoc]._.cluster_ids[0] <= cluster_id:
                maxdoc += 1
            while mindoc < len(self.docs) - 1 and self.docs[mindoc]._.cluster_ids[1] < cluster_id:
                mindoc += 1
            cluster = []
            
            for offset, doc in enumerate(self.docs[mindoc:maxdoc]):
                for (start, end, text, cl) in doc._.coref_ents:
                    if cluster_id == cl:
                        cluster.append((mindoc+offset, start, end, text))
            
            clusters.append((cluster, mindoc, maxdoc))
        self.clusters = clusters
        return clusters
    
    
    
        
    def saveWeights(self, out_file):
        G = name_unification_graph.CharacterUnificationGraph(self.docs, self.clusters).createGraph()
        with open(out_file, 'w') as f:
            for edge in self.graphToList(G):
                f.write(','.join([str(num) for num in edge]) + '\n')
        
    
    def graphToList(self, G):
        """
        For each edge, returns weights to predict if the nodes
        are connected or not.
        """
        edges = []
        for u, v, data in G.edges(data=True):
            edge_data = [u, v] + [int(data[key]) for key in data]
            edges.append(edge_data)
        return edges

    def predictProba(self, G, model_path):
        model = name_unification_model.getModel(model_path)
        weights = self.graphToList(G)
        
        prediction = model.predict_proba([w[2:] for w in weights])
        
        edge_prediction = dict((zip([(w[0], w[1]) for w in weights], [p[1] for p in prediction])))
        
        return edge_prediction

    def initFinalGraph(self, G, edge_prediction):
        final_graph = nx.Graph()
        final_graph.add_nodes_from(G.nodes)
        
        for (u, v), prob in edge_prediction.items():
            G[u][v]['prob'] = prob
            if prob > 0.5:
                final_graph.add_edge(u, v, prob=prob)
        return final_graph
    
    def removeNodes(self, final_graph, G, edge_prediction, edge_maxprob, character_remove_limit):
        for c in (nx.connected_components(final_graph)):
            if len(c) > 2:
                sorted_toremove = []
                for name_A in c:
                    for name_B in c:
                        if not (name_A, name_B) in edge_prediction:
                            continue
                        this_prob = edge_prediction[(name_A, name_B)]
                        if this_prob < 0.1:
                            sorted_toremove.append((this_prob, (name_A, name_B)))
                sorted_toremove.sort(key=lambda t: t[0])
                for prob, (name_A, name_B) in sorted_toremove:
                    try:
                        path = nx.shortest_path(final_graph, source=name_A, target=name_B)
                    except nx.NetworkXNoPath:
                        continue
                    lowest_prob = (edge_maxprob, None)
                    for i in range(len(path)-1):
                        curr_prob = final_graph[path[i]][path[i+1]]['prob']
                        if curr_prob < lowest_prob[0]:
                            lowest_prob = (curr_prob, i)
                    prob, i = lowest_prob
                    if i == None:
                        continue
                    final_graph.remove_edge(path[i], path[i+1])
        
        nodes_to_remove = []
        for c in (nx.connected_components(final_graph)):
            occurences = 0
            for node in c:
                occurences += G.nodes[node]['occurences']
            if occurences < character_remove_limit:
                nodes_to_remove += list(c)
        final_graph.remove_nodes_from(nodes_to_remove)
        return final_graph

    def renameNodes(self, final_graph, G, edge_prediction):
        narrator_nodes = [n for n, d in G.nodes(data=True) if d['type'] == "NARRATOR" and n in final_graph.nodes]
        
        occurences = [G.nodes[node]['occurences'] for node in narrator_nodes]
        if len(narrator_nodes) > 1:
            for node in narrator_nodes[1:]:
                G = nx.algorithms.minors.contracted_nodes(G, narrator_nodes[0], node)
                final_graph = nx.algorithms.minors.contracted_nodes(final_graph, narrator_nodes[0], node)
        if occurences and sum(occurences) > len(self.docs):
            mapping = {narrator_nodes[0]: "(THE NARRATOR)"}
            G = nx.relabel_nodes(G, mapping)
            final_graph = nx.relabel_nodes(final_graph, mapping)
            G.nodes["(THE NARRATOR)"]['occurences'] = sum(occurences)
            final_graph.nodes["(THE NARRATOR)"]['occurences'] = sum(occurences)
        elif narrator_nodes:
            G.remove_node(narrator_nodes[0])
            final_graph.remove_node(narrator_nodes[0])
        
        mapping = dict((node, data['name'] if data['name'] else node) for node, data in G.nodes(data=True) if data['type'] == "NAMELESS_CHAR")
        for short_name, correct_name in mapping.items():
            if short_name in final_graph.nodes:
                final_graph.nodes[short_name]['short_name'] = short_name
        final_graph = nx.relabel_nodes(final_graph, mapping)
        G = nx.relabel_nodes(G, mapping)
        return final_graph, G
        
    
    def smallGenderDifference(self, female, male):
        return abs(female-male) < min(female, male)/10
    
    
    def makeCharacters(self, final_graph, G):
        self.characters = {}
        char_id = 0
        for c in (nx.connected_components(final_graph)):
            variants = []
            final_gender_m, final_gender_f = 0, 0
            for name in c:
                occurences = G.nodes[name]['occurences']
                variants.append((name, occurences))
                
                gender_f, gender_m = G.nodes[name]['female_coref'], G.nodes[name]['male_coref']
                gender = 'F' if gender_f > gender_m else 'M'
                if G.nodes[name]['type'] == "PERSON":
                    person = G.nodes[name]['person']
                    if person.gender and self.smallGenderDifference(gender_f, gender_m):
                        gender = person.gender or gender
                if gender == 'F':
                    final_gender_f += occurences
                else:
                    final_gender_m += occurences
            final_gender = 'F' if final_gender_f > final_gender_m else 'M'
            self.characters[char_id] = (variants, final_gender)
            char_id += 1
        
        return self.characters
        
    
    def markCharactersInCoref(self, characters):
        """
        For each coreference cluster, finds to which character it belongs.
        If a word is in more clusters, it chooses the first one (= the one that
        has most information before the word).
        
        If two different characters are in one cluster, nothing is chosen.
        """
        name_dict = {}
        gender_dict = {}
        for char_id, (variants_list, gender) in characters.items():
            gender_dict[char_id] = gender
            for name, count in variants_list:
                name_dict[name] = (char_id, gender)
        
        for (cluster, mindoc, maxdoc) in self.clusters:
            char_ids = []
            genders = []
            for (doc_id, start, end, text) in cluster:
                span = self.docs[doc_id][start:end]
                if text in name_dict:
                    (char_id, gender) = name_dict[text]
                    char_ids.append(char_id)
                    genders.append(gender)
                elif span.root in [e.root for e in self.docs[doc_id].ents]:
                    ent_i = [e.root for e in self.docs[doc_id].ents].index(span.root)
                    ent = self.docs[doc_id].ents[ent_i]
                    name_index = None
                    if ent.label_ == "NAMELESS_CHAR":
                        name_index = ent._.nameless_name
                    elif ent.label_ == "NARRATOR":
                        name_index = "(THE NARRATOR)"
                    if name_index and name_index in name_dict:
                        (char_id, gender) = name_dict[name_index]
                        char_ids.append(char_id)
                        genders.append(gender)
            
            if not char_ids:
                continue
            
            
            counts = Counter(char_ids).most_common()
            for (doc_id, start, end, text) in cluster:
                token = self.docs[doc_id][start:end].root
                gender = None
                if token.lower_ in self.gendered_words:
                    gender = self.gendered_words[token.lower_]
                    if not gender in ['F', 'M']:
                        gender = None
                found = False
                final_id = None
                for (proposed_id, count) in counts:
                    if not gender or gender == gender_dict[proposed_id]:
                        found = True
                        final_id = proposed_id
                        break
                if found:
                    if not token._.char_id:
                        token._.char_id = counts[0][0]
                if token.lower_ in ['his', 'her'] and self.docs[doc_id][token.i+1]._.char_id:
                    pass
                    
                
        for doc in self.docs:
            for ent in doc.ents:
                if ent.text in name_dict:
                    ent.root._.char_id = (name_dict[ent.text][0])
        
        for doc in self.docs:
            for token in doc:
                if token.text in name_dict:
                    token._.char_id = (name_dict[token.text][0])

    def setExtensions():
        if not Token.has_extension("char_id"):
            Token.set_extension("char_id", default=None)


class FalseCharacterExtractor(CharacterExtractor):
    """
    A class to also add character id to tokens in docs,
    but with the list of golden characters, to evaluate
    quote attribution on this list
    """
    def __init__(self, docs, characters):
        CharacterExtractor.__init__(self, docs)
        self.false_characters = characters
        self.character_names = {}
        for char_id, (names, gender) in characters.items():
            for (name, count) in names:
                self.character_names[name] = char_id
    def extractCharacters(self, model=None, edge_maxprob=None, character_remove_limit=None):
        self.reconstructClusters()
        self.markCharactersInCoref(self.false_characters)
        return self.false_characters


