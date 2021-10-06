from spacy.tokens import Doc, Span
import logging

class QuoteMention:
    def __init__(self, character_set, docs):
        self.character_set = character_set
        self.docs = docs
            
        return
    
    def setExtensions():
        Doc.set_extension("mention", default=None, force=True)
        Doc.set_extension("mention_sieve", default=None, force=True)
        
    def solveMentions(self):
        logging.info("Linking mentions...")
        sieves = []
        sieves.append(TrigramMatch(self.docs, self.character_set))
        sieves.append(PreviousParDetermined(self.docs, self.character_set))
        sieves.append(CommonSpeechVerb(self.docs, self.character_set))
        sieves.append(SingleMention(self.docs, self.character_set))
        sieves.append(PreviousVocativeDetection(self.docs, self.character_set))
        sieves.append(FinalMention(self.docs, self.character_set))
        sieves.append(ConversationalPattern(self.docs, self.character_set))
        sieves.append(LooseConversationalPattern(self.docs, self.character_set))
        
        for sieve in sieves:
            sieve()
        return

class MentionSieve:
    def __init__(self, docs, character_set):
        self.docs = docs
        self.character_set = character_set
        self.pronouns_nominative = ['he', 'she', 'i']
        self.pronouns_other = ['his', 'him', 'her']
        self.common_mentions = ['man', 'woman', 'girl', 'boy', 'wife', 'husband', 'brother', 'sister']
        self.speech_verbs = [   'say',
                                'cry',
                                'reply',
                                'add',
                                'think',
                                'observe',
                                'call',
                                'answer',
                                'whisper',
                                'shout',
                                'sigh',
                                'ask',
                                'mutter',
                                'repeat']

    def __call__(self):
        for i, doc in enumerate(self.docs):
            if not doc._.mention and doc._.quotes:
                self.run(doc, i)
                if doc._.mention:
                    doc._.mention_sieve = self.sieve_name
    
    def isMention(self, doc, span):
        if len(span) == 1:
            if span[0].lower_ in self.pronouns_nominative:
                return True
            if span[0].lower_ in self.common_mentions:
                return True
        if span.text in self.character_set:
            return True
        for (start, end, _, _) in doc._.coref_ents:
            if span.start == start and span.end == end and not (span.text.lower() in self.pronouns_other):
                return True
        #if not span.root._.char_id == None:
        #    return True
        return False
    
    def isSpeechVerb(self, token):
        if token.pos_ == "VERB":
            if not token._.is_direct_speech:
                if token.lemma_.lower() in self.speech_verbs:
                    return True
        return False
    
    def getSubjIndices(self, root, passive_too=True):
        if not (root.pos_ == "VERB" or root.pos_ == "AUX"):
            return None
        subject = []
        while True:
            for child in root.children:
                if child.dep_ == "nsubj" or (passive_too and child.dep_ == "nsubjpass"):
                    for compound in child.children:
                        if compound.dep_ == "compound":
                            subject.append(compound)
                    subject.append(child)
            if subject or not root.dep_ == "conj":
                break
            root = root.head
        
        
        if not subject:
            return None
        
        subject_start = min([t.i for t in subject])
        subject_end = max([t.i for t in subject]) + 1
        
        return (subject_start, subject_end)
    
    
    def getAllMentions(self, doc):
        all_mentions = []
        for token in doc:
            if token._.is_direct_speech:
                continue
            if token.pos_ == "VERB" or token.pos_ == "AUX":
                subj = self.getSubjIndices(token)
                if subj and self.isMention(doc, Span(doc, subj[0], subj[1])):
                    all_mentions.append(subj)
        return all_mentions
    
    def getReallyAllMentions(self, doc):
        all_mentions = []
        for token in doc:
            if token._.is_direct_speech:
                continue
            if token._.char_id:
                all_mentions.append((token.i, token.i+1))
        return all_mentions
    
    

class TrigramMatch(MentionSieve):
    def __init__(self, docs, character_set):
        MentionSieve.__init__(self, docs, character_set)
        self.sieve_name = "trigramMatch"
    
    def run(self, doc, i):
        for q_start, q_end in doc._.quotes:
            if q_end < len(doc) - 1:
                after_1 = doc[q_end]
                after_2 = doc[q_end+1]
                # Quote-Mention-Verb
                if self.isMention(doc, doc[after_1.i:after_1.i+1]):
                    if self.isSpeechVerb(after_2):
                        doc._.mention = ((i, (q_end, q_end+1)))
                # Quote-Verb-Mention
                if self.isMention(doc, doc[after_2.i:after_2.i+1]):
                    if self.isSpeechVerb(after_1):
                        doc._.mention = ((i, (q_end+1, q_end+2)))
            if q_start > 1:
                before_1 = doc[q_start-1]
                before_2 = doc[q_start-2]
                # Mention-Verb-Quote
                if self.isMention(doc, doc[before_1.i:before_1.i+1]):
                    if self.isSpeechVerb(before_2):
                        doc._.mention = ((i, (q_start-1, q_start)))
                # Verb-Mention-Quote
                if self.isMention(doc, doc[before_2.i:before_2.i+1]):
                    if self.isSpeechVerb(before_1):
                        doc._.mention = ((i, (q_start-2, q_start-1)))
        return


class PreviousParDetermined(MentionSieve):
    def __init__(self, docs, character_set):
        MentionSieve.__init__(self, docs, character_set)
        self.sieve_name = "previousParDetermined"
    
    def run(self, doc, i):
        if i <= 0:
            return
        prev_doc = self.docs[i-1]
        if prev_doc.text.endswith(":") or prev_doc.text.endswith("--") or prev_doc.text.endswith(","):
            verbs = [list(prev_doc.sents)[-1].root]
            for verb in prev_doc:
                if (verb.pos_ == "VERB" or verb.pos_ == "AUX")  and verb.dep_ == "conj" and verb.head in verbs:
                    verbs.append(verb)
            
            for verb in verbs:
                subj = self.getSubjIndices(verb)
                if subj and self.isMention(prev_doc, Span(prev_doc, subj[0], subj[1])):
                    doc._.mention = (i-1, subj)
        return


class CommonSpeechVerb(MentionSieve):
    def __init__(self, docs, character_set):
        MentionSieve.__init__(self, docs, character_set)
        self.sieve_name = "commonSpeechVerb"
    
    def run(self, doc, i):
        for token in doc:
            if self.isSpeechVerb(token):
                subj = self.getSubjIndices(token, passive_too=False)
                if subj:
                    doc._.mention = (i, subj)
                    return
        return


class SingleMention(MentionSieve):
    def __init__(self, docs, character_set):
        MentionSieve.__init__(self, docs, character_set)
        self.sieve_name = "singleMention"
    
    def run(self, doc, i):
        all_mentions = self.getReallyAllMentions(doc)
        if len(all_mentions) == 1:
            doc._.mention = (i, all_mentions[0])
        elif all_mentions:
            first_mention = doc[all_mentions[0][0]:all_mentions[0][1]]
            for men in all_mentions:
                char_id = doc[men[0]:men[1]].root._.char_id
                if not char_id == first_mention.root._.char_id:
                    return
            doc._.mention = (i, all_mentions[0])
        return


class PreviousVocativeDetection(MentionSieve):
    def __init__(self, docs, character_set):
        MentionSieve.__init__(self, docs, character_set)
        self.sieve_name = "previousVocativeDetection"
    
    def run(self, doc, i):
        if i <= 0:
            return
        prev_doc = self.docs[i-1]
        all_vocatives = self.extractVocatives(prev_doc)
        if all_vocatives:
            doc._.mention = (i-1, all_vocatives[-1])
        return
    
    def extractVocatives(self, doc):
        people_addressed = []
        for sent in doc.sents:
            separators = [token.i for token in sent if (token.text in [',', ';', '?', '!']) and sent.end-1 > token.i]
            parts = []
            if separators:
                parts.append(Span(doc, max(0, sent.start), separators[0]))
                for i in range(1, len(separators)):
                    parts.append(Span(doc, separators[i-1]+1, separators[i]))
                end = sent.end-1 if doc[sent.end-1].is_punct else sent.end
                parts.append(Span(doc, separators[-1]+1, end))
            for part in parts:
                if part._.is_direct_speech:
                    addr = self.getVocative(part)
                    if addr:
                        people_addressed.append(addr)
        return people_addressed
    
    def getVocative(self, span):
        if span.text in self.character_set:
            return span.start, span.end
        if len(span) >= 2 and span[1:].text in self.character_set:
            if span[0].lower_ == "my" or span[0].pos_ == "ADJ":
                return span.start, span.end
        if len(span) >= 3 and span[2:].text in self.character_set:
            if span[0].lower_ == "my" and span[1].pos_ == "ADJ":
                return span.start, span.end
        return None


class FinalMention(MentionSieve):
    def __init__(self, docs, character_set):
        MentionSieve.__init__(self, docs, character_set)
        self.sieve_name = "finalMention"
    
    def run(self, doc, i):
        if not doc.text.endswith('"'):
            return
    
        all_mentions = self.getAllMentions(doc)
        
        if not all_mentions:
            return
        selected_mention = all_mentions[0]
        
        (q_start, q_end) = doc._.quotes[-1]
        for m, (m_start, m_end) in enumerate(all_mentions):
            if m_start < q_start:
                selected_mention = all_mentions[m]
            else:
                break
        
        doc._.mention = (i, selected_mention)
        return


class ConversationalPattern(MentionSieve):
    def __init__(self, docs, character_set):
        MentionSieve.__init__(self, docs, character_set)
        self.sieve_name = "conversationalPattern"
    
    def run(self, doc, i):
        if i <= 1:
            return
        doc_prev = self.docs[i-1]
        doc_before_prev = self.docs[i-2]
        if doc_prev[:]._.is_direct_speech and doc[:]._.is_direct_speech:
            doc._.mention = doc_before_prev._.mention
        return


class LooseConversationalPattern(MentionSieve):
    def __init__(self, docs, character_set):
        MentionSieve.__init__(self, docs, character_set)
        self.sieve_name = "looseConversationalPattern"
    
    def run(self, doc, i):
        if i <= 1:
            return
        doc_before_prev = self.docs[i-2]
        if doc_before_prev._.mention:
            (j, (s, e)) = doc_before_prev._.mention
            if j < i-2:
                return
        doc._.mention = doc_before_prev._.mention
        return



