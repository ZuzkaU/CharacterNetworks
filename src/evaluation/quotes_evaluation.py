import xml.etree.ElementTree as ET
from collections import Counter
import networkx as nx
from networkx.algorithms import bipartite
from spacy.tokens import Doc


class QuotesEvaluator:
    def __init__(self, docs, golden, characters):
        self.docs = docs
        self.golden = golden
        self.characters = characters
        
        self.char_dict = {}
        for char_id, (variants, gender) in characters.items():
            most_used_name, most_count = variants[0]
            for (name, count) in variants:
                if count > most_count:
                    most_used_name = name
            self.char_dict[char_id] = most_used_name
    
    def buildGraph(self, predicted, gold_speakers):
        
        B = nx.Graph()
        # predicted are ints, golden are strings -> nodes are different
        B.add_nodes_from([p for p in predicted if not p == None], bipartite=0)
        B.add_nodes_from([g for g in gold_speakers if not g == None], bipartite=1)
        
        for p in set(predicted):
            for g in set(gold_speakers):
                if not p == None and not g == None:
                    B.add_edge(p, g, weight=0)
        
        for p, g in zip(predicted, gold_speakers):
            if not p == None and not g == None:
                # subtracting, because we will find minimum weight matching
                B[p][g]['weight'] -= 1
        
        return B
    
    def getMatching(self, B):
        matches = bipartite.minimum_weight_full_matching(B)
        return matches
        
    def getAccuracy(self, B, matches, gold_speakers):
        correct_attributions = 0
        for match in matches:
            if type(match) == int:
                gold_node = matches[match]
                correct_attributions -= B[match][gold_node]['weight']
        
        num_gold = len([g for g in gold_speakers if g and not (g == "UNSURE" or g == "NOTANUTTERANCE")])
        
        accuracy = correct_attributions / num_gold
        #self.printResults(matches, gold_speakers)
        return accuracy
    
    def getPR(self, B, matches, gold_speakers, predicted):
        correct_attributions = 0
        all_attributions = 0
        all_gold = 0
        for gold, pred in zip(gold_speakers, predicted):
            if gold in matches and matches[gold] == pred:
                correct_attributions += 1
                all_attributions += 1
                all_gold += 1
            elif not pred == None and gold:
                all_attributions += 1
                all_gold += 1
            elif pred == None and gold:
                all_gold += 1
        
        precision = correct_attributions / all_attributions
        recall = correct_attributions / all_gold
        return precision, recall
        
    def printResults(self, matches, gold_speakers):
        
        for match in matches:
            if type(match) == int:
                mine = self.char_dict[match]
                their = matches[match]
                print(mine, "(" + str(match) + ") ---", their)
        
        
        with open('current-results.txt', 'w') as f:
        
            type_dict = {'explicit':(0, 0), 'anaphoric (pronoun)':(0, 0), 'anaphoric (other)':(0, 0), 'implicit':(0, 0)}
            for (i, doc), gold_sp in zip(enumerate(self.docs), gold_speakers):
                
                f.write(doc.text + '\n')
                
                mention = None
                context = None
                mention_sieve = None
                speaker = None
                speaker_sieve = None
                speaker_type = None
                true_speaker = None
                result = None
                
                if doc._.mention:
                    (d, (s, e)) = doc._.mention
                    if self.docs[d][s:e].text.lower() in ['he', 'she']:
                        speaker_type = 'anaphoric (pronoun)'
                    elif self.docs[i]._.speaker_sieve == "exactNameMatch":
                        speaker_type = 'explicit'
                    else:
                        speaker_type = 'anaphoric (other)'
                    if not d == i:
                        speaker_type = 'implicit'
                    
                    mention = self.docs[d][s:e].text
                    start = s-3 if s-3 > 0 else 0
                    end = e + 3 if e+3 < len(self.docs[d]) else len(self.docs[d])
                    context = self.docs[d][start:end].text
                    mention_sieve = self.docs[i]._.mention_sieve
                else:
                    speaker_type = "implicit"
                if doc[:]._.is_direct_speech:
                    speaker_type = 'implicit'
                if doc._.speaker_id:
                    speaker = self.char_dict[doc._.speaker_id]
                    speaker_sieve = self.docs[i]._.speaker_sieve
                true_speaker = gold_sp
                
                if gold_sp:
                    if gold_sp in matches:
                        true_id = matches[gold_sp]
                        if doc._.speaker_id == true_id:
                            result = "TRUE"
                    if doc._.speaker_id == None:
                        result = "NONE"
                    if not result:
                        result = "FALSE"
                    
                    a, b = (1 if result == "TRUE" else 0, 1)
                    c, d = type_dict[speaker_type]
                    type_dict[speaker_type] = (a+c, b+d)
                if True or (result == "FALSE" or result == "NONE"):
                    f.write("mention:" + str(mention) + '\n')
                    f.write("context:" + str(context) + '\n')
                    f.write("mention_sieve:" + str(mention_sieve) + '\n')
                    f.write("speaker_sieve:" + str(speaker_sieve) + '\n')
                    f.write("speaker:" + str(speaker) + '\n')
                    f.write("true_speaker:" + str(true_speaker) + '\n')
                    f.write("speaker_type:" + str(speaker_type) + '\n')
                    f.write("result:" + str(result) + '\n')
                    if doc._.quotes:
                        f.write("SPEAKERS:" + '\n')
                        for speaker_id, count in doc._.speakers_list:
                            f.write('\t' + self.char_dict[speaker_id] + str(count) + '\n')
                #elif result == "TRUE":
                #    f.write("speaker_sieve:" + str(speaker_sieve) + '\n')
                f.write('\n')
            
        total = 0
        correct = 0
        for t, counts in type_dict.items():
            print(t, counts, counts[0]/counts[1])
            total += counts[1]
            correct += counts[0]
        print(total, correct, correct/total)
        return
    
    def checkTextsSame(self, gold_data, docs):
        if not len(docs) == len(gold_data):
            print("GOLD:", len(gold_data), "DOCS:", len(docs))
            raise Exception("Golden data not equal to docs (different length)")
        for (gold_text, gold_speaker), doc in zip(gold_data, docs):
            if not gold_text == doc.text:
                print("GOLD:", gold_text)
                print("DOC:", doc.text)
                raise Exception("Golden data not equal to docs")


class QuotesEvaluatorQuoteLi3(QuotesEvaluator):
    def addGoldSpeakers(self):
        gold_data = self.parse(self.golden)
        if not Doc.has_extension('gold_speaker'):
            Doc.set_extension('gold_speaker', default=None)
        if not Doc.has_extension('gold_match_id'):
            Doc.set_extension('gold_match_id', default=None)
        self.checkTextsSame(gold_data, self.docs)
        for (gold_text, gold_speaker), doc in zip(gold_data, self.docs):
            doc._.gold_speaker = gold_speaker
        
        predicted = [doc._.speaker_id for doc in self.docs]
        gold_speakers = [speaker for (_, speaker) in gold_data]
        B = self.buildGraph(predicted, gold_speakers)
        matching = self.getMatching(B)
        
        for doc in self.docs:
            gold = doc._.gold_speaker
            if not gold or not gold in matching:
                continue
            match = matching[gold]
            if -B[match][gold]['weight'] <= 1:
                doc._.gold_match_id = None
            else:
                doc._.gold_match_id = match
        
        return self.docs
    
    
    def evaluate(self):
        gold_data = self.parse(self.golden)
        
        gold_texts = [text for (text, _) in gold_data]
        gold_speakers = [speaker for (_, speaker) in gold_data]
        
        self.checkTextsSame(gold_data, self.docs)
        
        predicted = [doc._.speaker_id for doc in self.docs]
        B = self.buildGraph(predicted, gold_speakers)
        matching = self.getMatching(B)
        accuracy = self.getAccuracy(B, matching, gold_speakers)
        
        return accuracy
            
    def evaluatePR(self):
        gold_data = self.parse(self.golden)
        
        gold_texts = [text for (text, _) in gold_data]
        gold_speakers = [speaker for (_, speaker) in gold_data]
        
        self.checkTextsSame(gold_data, self.docs)
        
        predicted = [doc._.speaker_id for doc in self.docs]
        B = self.buildGraph(predicted, gold_speakers)
        matching = self.getMatching(B)
        return self.getPR(B, matching, gold_speakers, predicted)
    

    def parse(self, golden_file):
        tree = ET.parse(golden_file)
        root = tree.getroot()
        characters = root.find('characters')
        text = root.find('text')
        chapters = text.findall('chapter')
        new_text = ['<text>']
        for chapter in chapters:
            lines = [l for l in ET.tostring(chapter, encoding="unicode").splitlines() if l]
            paragraphs = []
            for i, line in enumerate(lines):
                if i == 0:
                    assert line.startswith('<chapter>')
                    paragraphs.append('<chapter><par>' + line[len('<chapter>'):] + '</par>')
                elif i == len(lines) - 1:
                    assert line.endswith('</chapter>')
                    paragraphs.append('<par>' + line[:-len('</chapter>')] + '</par></chapter>')
                else:
                    paragraphs.append('<par>' + line + '</par>')
            new_text.append('\n'.join(paragraphs))
        new_text.append('</text>')
        
        gold_data = []
        new_root = ET.fromstring('\n'.join(new_text))
        new_text = []
        i = 0
        for chapter in new_root:
            for par in chapter:
                new_text.append(''.join(par.itertext()))
                all_speakers = []
                for quote in par.findall('quote'):
                    all_speakers.append(quote.attrib['speaker'])
                counts = Counter(all_speakers).most_common()
                if ''.join(par.itertext()) == '':
                    continue
                if len(counts) > 0:
                    majority_speaker, _ = counts[0]
                    gold_data.append((''.join(par.itertext()), majority_speaker))
                else:
                    gold_data.append((''.join(par.itertext()), None))
                i += 1
        return gold_data



class QuotesEvaluatorCQSC(QuotesEvaluator):
    def evaluate(self):
        self.parse()
        predicted = [doc._.speaker_id for doc in self.docs]
        self.buildGraph(predicted, self.gold_speakers)
        
        return self.precision, self.recall


    def parse(self):
        tree = ET.parse(self.golden)
        root = tree.getroot()
        
        
        if not len(root) == len(self.docs):
            print("lengths not equal", len(root), len(self.docs))
            return
        
        CQSC_persons = {}
        CQSC_entities = {}
        for person in root.iter('PERSON'):
            CQSC_persons[person.attrib['id']] = person.attrib['entity']
            if person.attrib['entity'] in CQSC_entities:
                CQSC_entities[person.attrib['entity']].add(''.join(person.itertext()))
            else:
                CQSC_entities[person.attrib['entity']] = set([''.join(person.itertext())])
        
        self.gold_speakers = []
        for child, doc in zip(root, self.docs):
            if not ''.join(child.itertext()) == doc.text:
                print(''.join(child.itertext()))
                print(doc.text)
                print()
                print("texts not equal")
                return
            speaker = None
            for grand in child:
                if grand.tag == "QUOTE":
                    if 'speaker' in grand.attrib:
                        if grand.attrib['speaker'] in CQSC_persons:
                            print(''.join(grand.itertext()))
                            print("Entity:", CQSC_entities[CQSC_persons[grand.attrib['speaker']]])
                            print(doc._.speaker_id)
                            speaker = CQSC_entities[CQSC_persons[grand.attrib['speaker']]]
                        print()
            self.gold_speakers.append(speaker)





