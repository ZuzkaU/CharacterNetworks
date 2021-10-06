import spacy
from spacy import Language
from spacy.tokens import Doc, Span, Token


@Language.factory("quote_parser")
def createQuoteParser(nlp, name):
    return QuoteParser()


class QuoteParser:
    def __init__(self):
        QuoteParser.setExtensions()
    
    def setExtensions():
        Token.set_extension("is_direct_speech", default=False)
        Span.set_extension("is_direct_speech", getter=
            lambda span: all([token._.is_direct_speech or token.is_quote for token in span]))
        Span.set_extension("contains_direct_speech", getter=
            lambda span: any([token._.is_direct_speech for token in span]))
        Span.set_extension("contains_undirect_speech", getter=
            lambda span: any([not token._.is_direct_speech and not (token.text == '"' or token.text == "'") for token in span]))
        Doc.set_extension("contains_direct_speech", getter=
            lambda doc: doc[0:len(doc)]._.contains_direct_speech)
        Doc.set_extension("quotes", default=[])
    
    def __call__(self, doc):
        doc = addDirectSpeechMarks(doc)
        return doc
        

def addDirectSpeechMarks(doc):
    is_direct = False
    current_quote_start = None
    for token in doc:
        token._.is_direct_speech = is_direct
        if '"' in token.text:
            if current_quote_start == None:
                current_quote_start = token.i
            else:
                doc._.quotes.append((current_quote_start, token.i+1))
                current_quote_start = None
            is_direct = not is_direct
            token._.is_direct_speech = False
        elif '“' in token.text:
            current_quote_start = token.i
            is_direct = True
            token._.is_direct_speech = False
        elif '”' in token.text:
            if not current_quote_start == None:
                doc._.quotes.append((current_quote_start, token.i+1))
            current_quote_start = None
            is_direct = False
            token._.is_direct_speech = False
    return doc
