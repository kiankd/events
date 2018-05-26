import simple_helpers as helpers
import xml.etree.ElementTree as ET
# from python.general_classes import MentionCoreferenceClusters


def is_not_action(xml_tag):
    return not (xml_tag.startswith('ACTION') or xml_tag.startswith('NEG_ACTION'))

class EcbDocument(object):
    def __init__(self, xml_file, path, topic, annotated_sentences, pmention_clusters, get_all=False, events_only=False):
        """
        :type pmention_clusters: MentionCoreferenceClusters
        """

        # organizational attributes
        self.fname = xml_file
        self.category = helpers.get_category(xml_file)
        self.topic = topic
        self.docnum = int(xml_file.split('_')[1].strip(self.category+'.xml'))
        self.sentences = {}
        self.mentions = {}

        # parsing constants initialization
        tokens_dict = {}
        relevant_tokens = {}

        # token parsing
        root = ET.parse(path+xml_file, parser=ET.XMLParser()).getroot()
        for tok in root.findall('token'):
            sentnum = int(tok.get('sentence'))
            if get_all or sentnum in annotated_sentences:
                tid = tok.get('t_id')
                token = EcbToken(tok.text, tid, sentnum)
                helpers.update_list_dict(tokens_dict, sentnum, token)
                relevant_tokens[tid] = token

        self.sentences = {sentnum:EcbSentence(token_list, sentnum, xml_file) for sentnum, token_list in tokens_dict.iteritems()}

        # get mentions
        for item in root.find('Markables'):
            tag = item.tag
            mid = item.get('m_id')

            if is_not_action(tag) and events_only:
                continue

            if item.attrib.has_key('TAG_DESCRIPTOR'):
                pmention_clusters.add_instance(RealInstance(tag, mid, item.get('TAG_DESCRIPTOR'), item.get('instance_id'), xml_file))
            else:
                orig_ids = [int(child.get('t_id')) for child in item]
                mention_tok_ids = map(str, range(orig_ids[0], orig_ids[-1]+1))

                if helpers.all_in_list(mention_tok_ids, relevant_tokens.keys()):
                    # old_len = len(corefed_token_ids)
                    # corefed_token_ids = corefed_token_ids.union(set(orig_ids))

                    # if len(corefed_token_ids) - old_len != len(orig_ids):
                    #     print 'OVERLAP - continuing'
                    #     continue

                    tokens = [relevant_tokens[tid] for tid in mention_tok_ids]
                    self.mentions[mid] = Mention(self.fname, mid, tag, tokens, is_continuous=orig_ids==mention_tok_ids)


        # get coreference
        mids_mapped = set()
        for coreference in root.find('Relations'):
            iid = coreference.get('note')
            if iid is None: #intra-doc-coref
                iid = helpers.get_intra_doc_iid(coreference.find('target').get('m_id'), xml_file)

            for child in coreference:
                mid = child.get('m_id')
                if child.tag == 'source' and mid in self.mentions.keys():
                    try:
                        pmention_clusters.add_mention(iid, self.mentions[mid])
                        mids_mapped.add(mid)
                    except KeyError:
                        pass

        for mid,mention in self.mentions.iteritems():
            if mid not in mids_mapped:
                pmention_clusters.add_singleton_mention(mention)

    def to_tokenized_file_string(self):
        s = ''
        for sentnum in sorted(self.sentences.keys()):
            sent = self.sentences[sentnum]
            s += sent.tokenized() + '\n'
        return s

    def itersentences(self):
        for sentence in self.sentences.itervalues():
            yield sentence

    def get_all_tokens(self):
        toks = []
        for sentence in self.itersentences():
            toks += sentence.tokens
        return toks


class EcbSentence(object):
    def __init__(self, tokens, sentnum, doc):
        self.tokens = tokens
        self.sentnum = sentnum
        self.doc = doc

    def __repr__(self):
        s = ''
        for token in self.tokens:
            s += token.__repr__() + ' '
        return s

    def tokenized(self):
        s = ''
        for token in self.tokens:
            s += token.text + ' '
        return s[:-1]


class EcbToken(object):
    def __init__(self, text, tid, sentnum):
        self.text = text.encode('utf-8')
        self.tid = tid
        self.sentnum = sentnum
        self.coref_ids = []
        self.coref_classes = []
        self.coref_idx = 0

    def __repr__(self):
        return self.text

    def set_coref_id(self, cid, cclass, is_last=None):
        self.coref_ids.append(cid)
        self.coref_classes.append(cclass)

    def get_coref_id(self):
        crt_coref_idx = self.coref_idx
        self.coref_idx += 1
        try:
            return self.coref_ids[crt_coref_idx]
        except IndexError:
            self.coref_idx -= 1
            return '-'


class RealInstance(object):
    def __init__(self, tag, mid, description, instance_id, xml_file):
        self.mclass = tag
        self.mid = mid
        self.description = description
        self.iid = instance_id
        self.fname = xml_file
        if instance_id is None:
            self.iid = helpers.get_intra_doc_iid(mid, xml_file)

    def __repr__(self):
        return self.mclass+':'+self.iid

    def get_class(self):
        return self.mclass.split('_')[0]


class Mention(object):
    MENTION_CLASSES = ['ACTION', 'LOC', 'HUMAN', 'NON', 'TIME']
    NON_EVENT_CLASSES = ['LOC', 'HUMAN', 'NON', 'TIME']

    def __init__(self, fname, mid, tag, tokens, is_continuous=True):
        self.fname = fname
        self.mid = mid
        self.mclass = tag
        self.tokens = tokens
        self.coref_chain_id = None
        self.is_singleton = False
        self.is_continuous = is_continuous

    def __repr__(self):
        return self.mclass + ':' + str(self.tokens)

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def empty_mention(cls, fname, mclass=None):
        m = Mention(fname, None, None, [])
        m.mclass = mclass
        return m

    @classmethod
    def get_comparator_function(cls):
        return lambda mention: (mention.fname, mention.get_token_id())

    def is_event(self):
        return 'ACTION' in self.mclass

    def topic(self):
        return helpers.get_topic(self.fname)

    def get_token_id(self):
        try:
            return self.tokens[0].tid
        except KeyError:
            return None

    def get_start_end_token_ids(self):
        try:
            return self.tokens[0].tid, self.tokens[-1].tid
        except KeyError:
            return None, None

    def get_class(self):
        c = self.mclass.split('_')[0]
        if c == 'NEG':
            return 'ACTION'
        return c

    def get_sentnum(self):
        try:
            return self.tokens[0].sentnum
        except IndexError:
            return None

    def get_specific_class(self):
        s = self.mclass
        s = (s.lstrip('NEG_') if s.startswith('NEG_') else s).split('_')[0]
        if s in self.MENTION_CLASSES:
            return s
        else:
            raise Exception('No class ERROR: class name gotten = %s'%s)

    def set_coref_id(self, cid, singleton=False, reset=False):
        if self.coref_chain_id is None or reset:
            self.coref_chain_id = cid
            self.is_singleton = singleton
            for token in self.tokens:
                token.set_coref_id(cid, self.get_specific_class(), is_last=token==self.tokens[-1])
        else:
            raise Exception("ERROR: Coref ID for this mention has already been set!")

    def corefers_with(self, mention2):
        return self.coref_chain_id == mention2.coref_chain_id
