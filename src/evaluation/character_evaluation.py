import networkx as nx
from networkx.algorithms import bipartite

class CharacterEvaluator:
    def __init__(self):
        pass
    
    def parseCharGender(char_file):
        with open(char_file, 'r') as f:
            characters = f.read().splitlines()
        gold_dict = {}
        for character in characters:
            parts = character.split(',')
            char_id = int(parts[0])
            if char_id in gold_dict:
                gold_dict[char_id][0].append((parts[1], 0))
            else:
                gold_dict[char_id] = ([(parts[1], 0)], parts[2])
        return gold_dict
    
    def getMatching(self, graph):
        matches = bipartite.minimum_weight_full_matching(graph)
        
        cumulative_weight = 0
        for match in matches:
            if match >= 0:
                weight = graph[match][matches[match]]['weight']
                cumulative_weight += weight
        return matches, cumulative_weight

class CharacterEvaluatorVala(CharacterEvaluator):
    def parseValaGold(self, char_file):
        with open(char_file, 'r') as f:
            characters = f.read().splitlines()
        gold_dict = {}
        for character in characters[1:]:
            parts = character.split(',')
            char_id = int(parts[0])
            if char_id in gold_dict:
                gold_dict[char_id].append(parts[1])
            else:
                gold_dict[char_id] = [parts[1]]
        return gold_dict
    
    def parseValaPredicted(self, char_file):
        with open(char_file, 'r') as f:
            characters = f.read().splitlines()
        pred_dict = {}
        for i, character in enumerate(characters):
            names = [name for j, name in enumerate(character.split(',')) if j % 2 == 0]
            pred_dict[i] = names
        return pred_dict
    
    def getPrecision(self, predicted, golden):
        Bp = nx.Graph()
        Bp.add_nodes_from(predicted.keys(), bipartite=0)
        Bp.add_nodes_from([-i-1 for i in golden.keys()], bipartite=1)
        
        for pred_k in predicted:
            for gold_k in golden:
                intersection = len(set(predicted[pred_k]) & set(golden[gold_k]))
                weight = (intersection/len(predicted[pred_k]))
                # minimum weight = predicted is a subset of golden, what we want
                Bp.add_edge(pred_k, -gold_k-1, weight=-weight)
        
        matches, cumulative_weight = self.getMatching(Bp)
        precision = -cumulative_weight / len(predicted)
        
        return precision
    
    def getRecall(self, predicted, golden):
        Br = nx.Graph()
        Br.add_nodes_from(predicted.keys(), bipartite=0)
        Br.add_nodes_from([-i-1 for i in golden.keys()], bipartite=1)
        
        for pred_k in predicted:
            for gold_k in golden:
                intersection = len(set(predicted[pred_k]) & set(golden[gold_k]))
                if intersection > 0:
                    weight = 1
                else:
                    weight = 0
                Br.add_edge(pred_k, -gold_k-1, weight=-weight)
        
        matches, cumulative_weight = self.getMatching(Br)
        recall = -cumulative_weight / len(golden)
        return recall


class CharacterEvaluatorImportance(CharacterEvaluator):
    """
    Character weights are attributed by their importance in the novel.
    We don't penalize not finding minor characters as much as not finding
    main characters.
    """
    def parseCountsPredicted(self, pred_file):
        with open(pred_file, 'r') as f:
            characters = f.read().splitlines()
        pred_dict = {}
        for i, character in enumerate(characters):
            names = []
            name = None
            for j, item in enumerate(character.split(',')):
                if j % 2 == 0:
                    name = item
                else:
                    names.append((name, int(item)))
            pred_dict[i] = names
        return pred_dict
    
    def parseCountsGold(self, char_file):
        with open(char_file, 'r') as f:
            characters = f.read().splitlines()
        gold_dict = {}
        for character in characters:
            parts = character.split(',')
            char_id = int(parts[0])
            if char_id in gold_dict:
                gold_dict[char_id].append((parts[1], int(parts[2])))
            else:
                gold_dict[char_id] = [(parts[1], int(parts[2]))]
        return gold_dict
    
    
    def getPrecision(self, predicted, golden):
        B = nx.Graph()
        B.add_nodes_from(predicted.keys(), bipartite=0)
        B.add_nodes_from([-i-1 for i in golden.keys()], bipartite=1)
        
        pred_weights, gold_weights = self.getWeightDicts(predicted, golden)
        
        for pred_k, pred_namelist in predicted.items():
            for gold_k, gold_namelist in golden.items():
                pred_names = [item[0] for item in pred_namelist]
                gold_names = [item[0] for item in gold_namelist]
                
                intersection = set(pred_names) & set(gold_names)
                #weight = sum([gold_weights[n] for n in intersection]) / sum([gold_weights[n] if n in gold_weights else pred_weights[n] for n in pred_names])
                if intersection:
                    weight = sum([gold_weights[n] for n in intersection])
                else:
                    weight = 0
                B.add_edge(pred_k, -gold_k-1, weight=-weight)
        
        matches, cumulative_weight = self.getMatching(B)
        precision = -cumulative_weight / sum([gold_weights[n] if n in gold_weights else pred_weights[n] for n in pred_weights])
        
        
        return precision
    
    def getWeightDicts(self, predicted, golden):
        pred_weights = {}
        for pred_k, pred_namelist in predicted.items():
            for name, weight in pred_namelist:
                pred_weights[name] = weight
                if name == "(THE NARRATOR)":
                    pred_weights[name] = 0
        gold_weights = {}
        for gold_k, gold_namelist in golden.items():
            for name, weight in gold_namelist:
                gold_weights[name] = weight
        return pred_weights, gold_weights
    
    def getRecall(self, predicted, golden):
        B = nx.Graph()
        B.add_nodes_from(predicted.keys(), bipartite=0)
        B.add_nodes_from([-i-1 for i in golden.keys()], bipartite=1)
        
        pred_weights, gold_weights = self.getWeightDicts(predicted, golden)
        
        for pred_k, pred_namelist in predicted.items():
            for gold_k, gold_namelist in golden.items():
                pred_names = [item[0] for item in pred_namelist]
                gold_names = [item[0] for item in gold_namelist]
                
                intersection = set(pred_names) & set(gold_names)
                if intersection:
                    weight = sum([gold_weights[n] for n in gold_names])
                else:
                    weight = 0
                B.add_edge(pred_k, -gold_k-1, weight=-weight)
        
        matches, cumulative_weight = self.getMatching(B)
        
        
        recall = -cumulative_weight / sum([gold_weights[n] for n in gold_weights])
        return recall

