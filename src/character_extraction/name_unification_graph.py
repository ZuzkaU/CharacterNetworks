from tqdm import tqdm
import networkx as nx
import string

import character_extraction.name_parser as name_parser

class CharacterUnificationGraph:
    def __init__(self, docs, clusters):
        self.docs = docs
        self.G = nx.Graph()
        self.clusters = clusters


    def createGraph(self):
        self.addNodes()
        parser = name_parser.NameParser([name for (name, data) in self.G.nodes(data=True) if data['type'] == 'PERSON'])
        name_dict = parser.parseNames()
        for name, data in self.G.nodes(data=True):
            if data['type'] == 'PERSON':
                data['person'] = name_dict[name]
        
        edge_dict = {
            'coref_connected'     : 0,
            'coref_unconnected'   : 0,
            'gender_same'         : 0,
            'gender_different'    : 0,
            'conjunction'         : 0,
            'honorific_differ'    : 0,
            'name_subset'         : 0,
            'first_name_variant'  : 0,
            'name_part_differ'    : 0,
            'same_substring'      : 0
        }
        for node_A in self.G.nodes(data=False):
            for node_B in self.G.nodes(data=False):
                if not node_A == node_B and not (node_A, node_B) in self.G.edges:
                    self.G.add_edge(node_A, node_B)
                    for key in edge_dict:
                        self.G[node_A][node_B][key] = 0
       
        self.addCoreferenceInfo()
        self.addNameVariantsEdges()
        self.addDependencyEdges()
        self.addGenderEdges()
        return self.G
        
    
    def containsWeirdCharacters(self, text):
        """
        returns True if the text contains other that alphabetic characters
        or "-" or "'" or " ",
        or if the text contains capital letters other than at beginning of words
        """
        for char in text:
            if not char in string.ascii_letters + "'" + "-" + " " + ".":
                return True
        if (len(text) > 2 and text[-2:] == "'s") or text.endswith("'"):
            return True
        text = text.replace('-', ' ')
        text = text.replace("'", ' ')
        text = text.replace(".", '')
        for part in text.split(' '):
            if len(part) == 0:
                return True
            for char in part[1:]:
                if not char in string.ascii_lowercase:
                    return True
        return False
    
    
    def addNodes(self):
        """
        Adds all Person and Unnamed_character entities to the graph.
        """
        ents = []
        for doc in self.docs:
            doc.set_ents([e for e in doc.ents if not self.containsWeirdCharacters(e.text)])
            ents += list(doc.ents)
        nodes = []
        for ent in ents:
            if ent.label_ == "PERSON":
                nodes.append((ent.text, {'type': ent.label_, 'occurences': 0, 'female_coref': 0, 'male_coref': 0, 'person': None}))
            elif ent.label_ == "NARRATOR":
                nodes.append((ent.text, {'type': ent.label_, 'occurences': 0, 'female_coref': 0, 'male_coref': 0, 'name': "(THE NARRATOR)"}))
            elif ent.label_ == "NAMELESS_CHAR":
                nodes.append((ent.text, {'type': ent.label_, 'occurences': 0, 'female_coref': 0, 'male_coref': 0, 'name': ent._.nameless_name}))
        self.G.add_nodes_from(nodes)
        
        if len(self.G.nodes) == 0:
            raise Exception("Could not find any characters!")
        
        for ent in ents:
            self.G.nodes[ent.text]['occurences'] += 1
    
    
    def addEdge(self, node1, node2, edge_type):
        if node1 == node2:
            return
        
        self.G[node1][node2][edge_type] += 1
    
    
    def addCoreferenceInfo(self):
        """
        Coreference clusters are saved separately for each document, and they
        can span multiple documents.
        
        We find all coreference entities that are also named entities (entA).
        A coreference entity can be in multiple clusters due to sliding window
        approach. For each cluster, we find all documents that participate in 
        the same cluster, and named coreference entities in this documents
        (entB). 
        
        If both entities are in the same cluster, we add an edge of type 
        'coref_connected', else of type 'coref_unconnected'.
        
        We also count coreference links to male or female pronouns.
        """
        for cluster_id, (cluster, mindoc, maxdoc) in tqdm(enumerate(self.clusters), desc="Connecting persons"):
            for coref_i, (A_doc_i, A_start, A_end, A_text) in enumerate(cluster):
                if not self.docs[A_doc_i][A_start:A_end].root in [e.root for e in self.docs[A_doc_i].ents]:
                    continue
                
                if not A_text in self.G.nodes:
                    A_text = self.docs[A_doc_i][A_start:A_end].root.text
                    if not A_text in self.G.nodes:
                        continue
                
                for B_doc_offset, B_doc in enumerate(self.docs[mindoc:maxdoc]):
                    # najít všechny entity v coreferenci a přidat podle toho, jestli jsou jako A nebo ne
                    for ent in B_doc.ents:
                        if cluster_id in ent.root._.clusters:
                            self.addEdge(A_text, ent.text, "coref_connected")
                        else:
                            self.addEdge(A_text, ent.text, "coref_unconnected")
                            
                
                for (_, _, _, pron_text) in cluster:
                    if pron_text in ['she', 'her']:
                        self.G.nodes[A_text]['female_coref'] += 1
                    elif pron_text in ['he', 'him', 'his']:
                        self.G.nodes[A_text]['male_coref'] += 1
        
        return                    



    def addNameVariantsEdges(self):
        """
        adds the following edge types:
        
        honorific_differ
        name_subset
        first_name_variant
        name_part_differ
        """
        # both pairs appear twice but it in fact does not matter
        for node_A, data_A in self.G.nodes(data=True):
            for node_B, data_B in self.G.nodes(data=True):
                if data_A['type'] == 'PERSON' and data_B['type'] == 'PERSON':
                    person_A = data_A['person']
                    person_B = data_B['person']
                    if person_A.honorificDiffer(person_B):
                        self.addEdge(node_A, node_B, 'honorific_differ')
                    if person_A.isSubsetOf(person_B) or person_B.isSubsetOf(person_A):
                        self.addEdge(node_A, node_B, 'name_subset')
                    if person_A.firstNamesVariant(person_B):
                        self.addEdge(node_A, node_B, 'first_name_variant')
                    if person_A.namePartDiffer(person_B):
                        self.addEdge(node_A, node_B, 'name_part_differ')
            if data_A['type'] == "NARRATOR" or data_B['type'] == "NARRATOR":
                continue
            if node_A.lower().startswith(node_B.lower()) or  node_B.lower().startswith(node_A.lower()):
                self.addEdge(node_A, node_B, 'same_substring')
            if node_A.lower().endswith(node_B.lower()) or  node_B.lower().endswith(node_A.lower()):
                self.addEdge(node_A, node_B, 'same_substring')
        return


    def addDependencyEdges(self):
        """
        adds the following edge types:
        
        conjunction
        """
        for doc in self.docs:
            for ent in doc.ents:
                if ent.root.dep_ == 'conj':
                    for other_ent in doc.ents:
                        if other_ent.root == ent.root.head:
                            self.addEdge(ent.text, other_ent.text, 'conjunction')
        return
    
    
    def addGenderEdges(self):
        """
        gender z coreference, ale i ze jména
        
        gender_same
        gender_different
        """
        def getGenders(female, male, person=None):
            genders = ['F' if female > male else 'M']
            if person and person.gender:
                genders.append(person.gender)
            return genders
        
        # both pairs appear twice but in fact it does not matter
        for node_A, data_A in self.G.nodes(data=True):
            for node_B, data_B in self.G.nodes(data=True):
                person_A = data_A['person'] if data_A['type'] == 'PERSON' else None
                genders_A = getGenders(data_A['female_coref'], data_A['male_coref'], person_A)
                
                person_B = data_B['person'] if data_B['type'] == 'PERSON' else None
                genders_B = getGenders(data_B['female_coref'], data_B['male_coref'], person_B)
                
                
                for gender_A in genders_A:
                    for gender_B in genders_B:
                        if gender_A == gender_B:
                            self.addEdge(node_A, node_B, 'gender_same')
                        else:
                            self.addEdge(node_A, node_B, 'gender_different')
        return
    

