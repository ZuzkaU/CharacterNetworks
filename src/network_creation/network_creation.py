import networkx as nx
import logging
from collections import Counter

class NetworkCreator:
    def __init__(self, docs, characters):
        self.docs = docs
        self.characters = characters
        
        self.name_dict = {}
        for char_id, (variants, gender) in self.characters.items():
            for name, count in variants:
                self.name_dict[name] = [char_id]
        
        return
    
    
    def createGenderNetwork(self):
        logging.info('Creating gender network...')
        G = self.initGraph()
        G = self.addConversationEdges(G)
        G = self.reduceGenderCharacters(G)
        G = self.renameNodes(G)
        logging.info('Gender network created.')
        return G
    
        
    def createNetworks(self, golden_speakers=False, node_descriptions=False):
        logging.info('Creating interaction networks...')
        conversationG = self.initGraph()
        cooccurrenceG = self.initGraph()
        if golden_speakers:
            goldConversationG = self.initGraph(gold=True)
        else:
            goldConversationG = None
        
    
        conversationG = self.addConversationEdges(conversationG)
        #self.reweightNodes(conversationG)
        cooccurrenceG = self.addCooccurenceEdges(cooccurrenceG)
        if golden_speakers:
            goldConversationG = self.addConversationEdges(goldConversationG, gold=True)
        
        if node_descriptions:
            descriptions = self.getNodeDescriptions()
            pass
        
        #conversationG, cooccurrenceG, goldConversationG = self.scaleWeights(conversationG, cooccurrenceG, goldConversationG)
        conversationG, cooccurrenceG, goldConversationG = self.reduceCharacters(conversationG, cooccurrenceG, goldConversationG)
            
        conversationG = self.renameNodes(conversationG)
        cooccurrenceG = self.renameNodes(cooccurrenceG)
        if golden_speakers:
            goldConversationG = self.renameNodes(goldConversationG, gold=True)
        
        logging.info('Interaction networks created.')
        return conversationG, cooccurrenceG, goldConversationG
        
    
    def initGraph(self, gold=False):
        G = nx.Graph()
        nodes = {}
        
        if not gold:
            for char_id, (variants, gender) in self.characters.items():
                most_common_name, count = None, 0
                total_count = 0
                for (name, name_count) in variants:
                    is_narrator = False
                    total_count += name_count
                    if name_count > count:
                        if name == "(THE NARRATOR)":
                            is_narrator = True
                        else:
                            most_common_name = name
                            count = name_count
                    if is_narrator and not most_common_name:
                        most_common_name = "(THE NARRATOR)"
                nodes[char_id] = (most_common_name, gender, total_count)
            G.add_nodes_from([(char_id, {'name': name, 'gender': gender, 'count': count, 'char_id':char_id}) for char_id, (name, gender, count) in nodes.items()])
        
        else:
            all_characters = {}
            self.gold_mapping = {}
            for doc in self.docs:
                if not doc._.gold_speaker == None:  
                    sp = doc._.gold_speaker
                    if sp in all_characters:
                        all_characters[sp] += 1
                    else:
                        all_characters[sp] = 1
                if not doc._.gold_match_id == None:
                    self.gold_mapping[doc._.gold_speaker] = doc._.gold_match_id
            
            G.add_nodes_from([(name, {'name': name, 'char_id': self.gold_mapping[name] if name in self.gold_mapping else None, 'count': count}) for name, count in all_characters.items()])
        
        for A in G.nodes:  
            for B in G.nodes:
                if A > B:
                    G.add_edge(A, B, weight=0)
        return G
    
    
    def addConversationEdges(self, G, gold=False):
        prev_speaker = None
        for doc in self.docs:
        
            if not gold:
                speaker = doc._.speaker_id
            else:
                speaker = doc._.gold_speaker
            
            if (not speaker == None) and (not prev_speaker == None) and (not speaker == prev_speaker):
                G[prev_speaker][speaker]['weight'] += 1
            prev_speaker = speaker
        
        return G
    
    def reweightNodes(self, G):
        node_weights = {}
        for n, data in G.nodes(data=True):
            node_weights[n] = 0
            for m in G.neighbors(n):
                node_weights[n] += G[m][n]['weight']
        
        for n, data in G.nodes(data=True):
            data['count'] = node_weights[n]
    
    def addCooccurenceEdges(self, G):
        for doc in self.docs:
            this_characters = set()
            for token in doc:
                if not token._.char_id == None:
                    this_characters.add(token._.char_id)
            for A in this_characters:
                for B in this_characters:
                    if A > B:
                        G[A][B]['weight'] += 1
        return G
    
    def reduceGenderCharacters(self, G):
        edges_to_remove = []
        for u, v, weight in G.edges.data('weight'):
            if weight == 0:
                edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)
        
        nodes_to_remove = []
        for c in nx.connected_components(G):
            if len(c) == 1:
                nodes_to_remove.append(list(c)[0])
        if len(nodes_to_remove) < len(G.nodes):
            G.remove_nodes_from(nodes_to_remove)
        
        most_frequent_characters = Counter()
        for node, data in G.nodes(data=True):
            most_frequent_characters += Counter({node: data['count'] + 1})
        most_frequent_characters = Counter(most_frequent_characters).most_common()
        if len(most_frequent_characters) > 20:
            nodes_to_remove = [n for (n, c) in most_frequent_characters[20:]]
            G.remove_nodes_from(nodes_to_remove)
        return G
        
        
    
    def reduceCharacters(self, convG, coocG, goldG=None):
        edges_to_remove = []
        for u, v, weight in convG.edges.data('weight'):
            if weight == 0:
                edges_to_remove.append((u, v))
        convG.remove_edges_from(edges_to_remove)
        
        edges_to_remove = []
        for u, v, weight in coocG.edges.data('weight'):
            if weight == 0:
                edges_to_remove.append((u, v))
        coocG.remove_edges_from(edges_to_remove)
        
        if not goldG:
            most_frequent_characters = Counter()
            for node, data in convG.nodes(data=True):
                most_frequent_characters += Counter({node: data['count'] + 1})
            most_frequent_characters = Counter(most_frequent_characters).most_common()
            if len(most_frequent_characters) > 20:
                nodes_to_remove = [n for (n, c) in most_frequent_characters[20:]]
                convG.remove_nodes_from(nodes_to_remove)
                coocG.remove_nodes_from(nodes_to_remove)
        
        if goldG:
            nodes_to_remove = [node for node, data in goldG.nodes(data=True) if data['char_id'] == None]
            goldG.remove_nodes_from(nodes_to_remove)
            
            edges_to_remove = []
            for u, v, weight in goldG.edges.data('weight'):
                if weight == 0:
                    edges_to_remove.append((u, v))
            goldG.remove_edges_from(edges_to_remove)
            
            
            remaining_nodes = [data['char_id'] for _, data in goldG.nodes(data=True)]
            nodes_to_remove = [node for node in convG.nodes if not node in remaining_nodes]
            convG.remove_nodes_from(nodes_to_remove)
            coocG.remove_nodes_from(nodes_to_remove)
        
        
        return convG, coocG, goldG
    
    def scaleWeights(self, convG, coocG, goldG=None):
        self.scaleWeight(convG)
        self.scaleWeight(coocG)
        if goldG:
            self.scaleWeight(goldG)
        return (convG, coocG, goldG)
    
    def scaleWeight(self, G):
        MAXWEIGHT = 200
    
        this_w = 0
        for u, v, data in G.edges(data=True):
            if data['weight'] > this_w:
                this_w = data['weight']
        
        mult = MAXWEIGHT / this_w
        
        for u, v, data in G.edges(data=True):
            data['weight'] *= mult
    
    
    def renameNodes(self, G, gold=False):
        if not gold:
            names = [(char_id, data['name']) for char_id, data in G.nodes(data=True)]
            mapping = dict(names)
            G = nx.relabel_nodes(G, mapping)
        else:
            pass
        return G
    
