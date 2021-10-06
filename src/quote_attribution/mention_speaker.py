from spacy.tokens import Doc, Span
import json
import logging
import os

class MentionSpeaker:
    def __init__(self, name_dict, gender_dict, docs):
        self.name_dict = name_dict
        self.docs = docs
        self.gender_dict = gender_dict
        
        self.gendered_words = {}
        vocab_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'vocab')
        with open(os.path.join(vocab_dir, 'gendered_words.json')) as genders_file:
            gendered_words_list = json.load(genders_file)
        for item in gendered_words_list:
            self.gendered_words[item['word']] = item['gender']
        
    
    def setExtensions():
        Doc.set_extension("speaker_id", default=None, force=True)
        Doc.set_extension("speaker_sieve", default=None, force=True)
        Doc.set_extension("speakers_list", default=None, force=True)
        Doc.set_extension("speakers_list_extended", default=None, force=True)
        
    
    def solveSpeakers(self):
        logging.info("Linking speakers...")
        
        self.makeSpeakersLists()
        
        sieves = []
        sieves.append(ExactNameMatch(self.docs, self.name_dict, self.gender_dict, self.gendered_words))
        sieves.append(Coreference(self.docs, self.name_dict, self.gender_dict, self.gendered_words))
        sieves.append(ConversationalPattern(self.docs, self.name_dict, self.gender_dict, self.gendered_words))
        #sieves.append(ClosestNameBefore(self.docs, self.name_dict, self.gender_dict, self.gendered_words))
        sieves.append(MajoritySpeaker(self.docs, self.name_dict, self.gender_dict, self.gendered_words))
        
        for sieve in sieves:
            sieve()
        
        return

    
    def distance(self, left, right):
        (left_doc_i, left_tok_i) = left
        (right_doc_i, right_tok_i) = right
        
        if left_doc_i == right_doc_i:
            return right_tok_i - left_tok_i
        elif left_doc_i > right_doc_i:
            return 0
        
        distance = 0
        distance += len(self.docs[left_doc_i]) - left_tok_i
        left_tok_i = 0
        left_doc_i += 1
        while left_doc_i < right_doc_i:
            distance += len(self.docs[left_doc_i])
            left_doc_i += 1
        distance += right_tok_i
        return distance
    
    
    def makeSpeakersLists(self):
        """
        For each doc containing direct speech, creates a list of possible
        speakers appearing around the doc.
        """
        mentions = []
        for i, doc in enumerate(self.docs):
            # save ALL mentions:
            for token in doc:
                if token._.char_id and token._.char_id != -1:
                    if not token._.is_direct_speech:
                        mentions.append((i, token.i))
        
        left, right = 0, 0
        for i, doc in enumerate(self.docs):
            # Make speakers list: count all coref identified ents 2000 tokens before and 500 tokens after the quote 
            # (possibly expansion to 2000 on both sides if nothing found is not done)
            if doc._.quotes:
                mintoken = (i, doc._.quotes[0][0])
                while left < len(mentions) - 1 and self.distance(mentions[left], mintoken) > 2000:
                    left += 1
                maxtoken = (i, doc._.quotes[-1][1])
                while right < len(mentions) and self.distance(maxtoken, mentions[right]) < 500:
                    right += 1
                
                speakers_dict = {}
                for (mention_doc, mention_token) in mentions[left:right]:
                    char_id = self.docs[mention_doc][mention_token]._.char_id
                    if self.docs[mention_doc][mention_token]._.is_direct_speech:
                        continue
                    if char_id == -1:
                        continue
                    dist = 1 + (min(self.distance((mention_doc, mention_token), mintoken), self.distance((mention_doc, mention_token), maxtoken)))
                    if char_id in speakers_dict:
                        speakers_dict[char_id] += 1# / dist
                    else:
                        speakers_dict[char_id] = 1# / dist
                speakers_list = sorted(speakers_dict.items(), key=lambda item: item[1])
                speakers_list.reverse()
                
                doc._.speakers_list = speakers_list
        return




class QuoteSieve:
    def __init__(self, docs, name_dict, gender_dict, gendered_words):
        self.docs = docs
        self.name_dict = name_dict
        self.gender_dict = gender_dict
        self.gendered_words = gendered_words
        

    def __call__(self):
        for i, doc in enumerate(self.docs):
            if doc._.speaker_id == None and doc._.quotes:
                self.run(doc, i)
            if (not doc._.speaker_id == None) and not doc._.speaker_sieve:
                doc._.speaker_sieve = self.sieve_name

    def getMentionGender(self, doc):
        gender = None
        if doc._.mention:
            (i, (start, end)) = doc._.mention
            span = self.docs[i][start:end]
            if span.text.lower() in self.gendered_words:
                gender = self.gendered_words[span.text.lower()].upper()
        if not gender in ['F', 'M']:
            gender = None
        return gender

    def isDiffThanNeighbor(self, doc_i, char_id):
        previous_speaker, next_speaker = None, None
        if doc_i >= 1:
            previous_speaker = self.docs[doc_i-1]._.speaker_id
        if doc_i < len(self.docs) - 1:
            next_speaker = self.docs[doc_i+1]._.speaker_id
        if previous_speaker == char_id:
            return False
        if next_speaker == char_id:
            return False
        return True


class ExactNameMatch(QuoteSieve):
    def __init__(self, docs, name_dict, gender_dict, gendered_words):
        QuoteSieve.__init__(self, docs, name_dict, gender_dict, gendered_words)
        self.sieve_name = "exactNameMatch"
    
    def run(self, doc, i):
        if not doc._.mention:
            return
        (m, (start, end)) = doc._.mention
        name = self.docs[m][start:end].text
        if name in self.name_dict:
            doc._.speaker_id = self.name_dict[name]
        elif name == 'I' and '(THE NARRATOR)' in self.name_dict:
            doc._.speaker_id = self.name_dict['(THE NARRATOR)']
        return


class Coreference(QuoteSieve):
    def __init__(self, docs, name_dict, gender_dict, gendered_words):
        QuoteSieve.__init__(self, docs, name_dict, gender_dict, gendered_words)
        self.sieve_name = "coreference"
    
    def run(self, doc, i):
        if not doc._.mention:
            return
        (i, (start, end)) = doc._.mention
        span = self.docs[i][start:end]
        coref_id = span.root._.char_id
        gender = self.getMentionGender(doc)
        if coref_id:
            if not gender or self.gender_dict[coref_id] == gender:
                doc._.speaker_id = coref_id
        return


class ConversationalPattern(QuoteSieve):
    def __init__(self, docs, name_dict, gender_dict, gendered_words):
        QuoteSieve.__init__(self, docs, name_dict, gender_dict, gendered_words)
        self.sieve_name = "conversationalPattern"
    
    def run(self, doc, i):
        if i <= 1:
            return
        prev_id = self.docs[i-2]._.speaker_id
        if prev_id:
            gender = self.getMentionGender(doc)
            if not gender or gender == self.gender_dict[prev_id]:
                if self.isDiffThanNeighbor(i, prev_id):
                    doc._.speaker_id = prev_id
        return


class ClosestNameBefore(QuoteSieve):
    def __init__(self, docs, name_dict, gender_dict, gendered_words):
        QuoteSieve.__init__(self, docs, name_dict, gender_dict, gendered_words)
        self.sieve_name = "closestNameBefore"
    
    def run(self, doc, i):
        gender = self.getMentionGender(doc)
        for before_i, before_doc in enumerate(self.docs[i:max(0, i-3):-1]):
            ents = list(before_doc.ents)
            ents.reverse()
            for ent in ents:
                ent_id = ent.root._.char_id
                if ent.root._.is_direct_speech:
                    continue
                if ent_id:
                    if self.docs[i - 1]._.speaker_id == ent_id:
                        continue
                    if i + 1 < len(self.docs) and self.docs[i + 1]._.speaker_id == ent_id:
                        continue
                    if not gender or self.gender_dict[ent_id] == gender:
                        doc._.speaker_id = ent_id
                        break
            if doc._.speaker_id:
                break
        return


class MajoritySpeaker(QuoteSieve):
    def __init__(self, docs, name_dict, gender_dict, gendered_words):
        QuoteSieve.__init__(self, docs, name_dict, gender_dict, gendered_words)
        self.sieve_name = "majoritySpeaker"
        self.conversationalSieve = ConversationalPattern(docs, name_dict, gender_dict, gendered_words)
    
    def run(self, doc, i):
        self.conversationalSieve.run(doc, i)
        if doc._.speaker_id:
            doc._.speaker_sieve = self.conversationalSieve.sieve_name
            return
        gender = self.getMentionGender(doc)
        for (char_id, _) in doc._.speakers_list:
            if not gender or self.gender_dict[char_id] == gender:
                doc._.speaker_id = char_id
                if self.isDiffThanNeighbor(i, char_id):
                    break
        return

