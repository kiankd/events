import simple_helpers as helpers
import xml.etree.ElementTree as ET
from ecb_doc_classes import EcbDocument

def element_child_text(element, childname):
    return element.find(childname).text

class CoreDocument(object):
    def __init__(self, path, xml_file):
        self.fname = xml_file
        self.sentences = []
        self.category = helpers.get_category(xml_file)
        self.topic = helpers.get_topic(xml_file)

        token_accum = 0 # because tid resets after each sentence, but we want it to do so only after each document
        root = ET.parse(path + xml_file, parser=ET.XMLParser()).getroot()

        for sentence in root.findall('document/sentences/sentence'):
            tokens = []
            sentnum = int(sentence.get('id'))
            for token in sentence.findall('tokens/token'):
                tokens.append( Token(
                    str(int(token.get('id')) + token_accum),
                    element_child_text(token, 'word'),
                    element_child_text(token, 'lemma'),
                    element_child_text(token, 'POS'),
                    element_child_text(token, 'NER'),
                    sentnum,
                    xml_file
                ))

            dependencies = []
            for depset in sentence.findall('dependencies'):
                if depset.get('type') == 'collapsed-ccprocessed-dependencies':
                    for dep in depset:
                        dependencies.append( Dependency(
                            dep.get('type'),
                            dep.find('governor').get('idx'),
                            dep.find('dependent').get('idx'),
                            token_accum
                        ))

            self.sentences.append(Sentence(tokens, element_child_text(sentence, 'parse'), dependencies))
            token_accum = int(tokens[-1].tid)

    def itersentences(self):
        for s in self.sentences:
            yield s

    def get_all_tokens(self):
        toks = []
        for s in self.itersentences():
            toks += s.tokens
        return toks

    def itertokens(self):
        for token in self.get_all_tokens():
            yield token

    def convert_tokens(self, ecbdoc, doc_mentions):
        """
        :type ecbdoc: EcbDocument
        :type doc_mentions: list
        """
        tokens = self.get_all_tokens()
        ecbtokens = ecbdoc.get_all_tokens()
        assert len(tokens) == len(ecbtokens)

        token_dict = {}
        for i in xrange(len(tokens)):
            # print '%s - %s : %s - %s'%(tokens[i].word,tokens[i].tid,ecbtokens[i].tid,ecbtokens[i].text)
            tokens[i].tid = ecbtokens[i].tid
            tokens[i].sentnum = ecbtokens[i].sentnum
            tokens[i].coref_ids = ecbtokens[i].coref_ids
            tokens[i].coref_classes = ecbtokens[i].coref_classes
            token_dict[tokens[i].tid] = tokens[i]
            tokens[i].coref_idx_update()

        for mention in doc_mentions:
            for i in xrange(len(mention.tokens)):
                mention.tokens[i] = token_dict[mention.tokens[i].tid]
                mention.tokens[i].set_is_last_token(mention.coref_chain_id ,(i == len(mention.tokens) - 1))


class Sentence(object):
    def __init__(self, tokens, parse_str, deps):
        self.tokens = tokens
        self.parse_tree = parse_str
        self.deps = deps

    def __repr__(self):
        s = ''
        for token in self.tokens:
            s += token.__repr__() + ' '
        return s[:-1]


class Token(object):
    def __init__(self, tid, word, lemma, pos, ner, sentnum, fname):
        self.tid = tid
        self.word = word.encode('utf-8')
        self.lemma = lemma
        self.pos = pos
        self.ner = ner
        self.sentnum = sentnum
        self.fname = fname
        self.coref_ids = []
        self.coref_classes = []
        self.is_last_token_in_mention = {}
        self.coref_idxs = {}

    def __repr__(self):
        return (self.word+'/'+self.pos)

    def set_coref_id(self, cid, cclass, is_last):
        self.coref_ids.append(cid)
        self.coref_classes.append(cclass)
        self.coref_idxs[cid] = 0
        self.set_is_last_token(cid, is_last)

    def not_mention(self):
        return len(self.coref_ids) == 0

    def set_is_last_token(self, mid, boolean):
        helpers.update_list_dict(self.is_last_token_in_mention, mid, boolean)

    def is_last_token_for_mid(self, mid):
        self.coref_idxs[mid] += 1
        return self.is_last_token_in_mention[mid][self.coref_idxs[mid] - 1]

    def reset_coreference(self):
        self.coref_classes = []
        self.coref_ids = []
        self.coref_idxs = {}
        self.is_last_token_in_mention = {}

    def coref_idx_update(self):
        for cid in self.coref_ids:
            self.coref_idxs[cid] = 0


class Dependency(object):
    def __init__(self, type_, gov, dep, token_accum):
        self.type = type_
        self.gov = str(int(gov)+token_accum)
        self.dep = str(int(dep)+token_accum)

empty_token = Token('','','','','',-1,'')
