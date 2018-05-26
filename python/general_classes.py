import simple_helpers as helpers
import numpy as np
from copy import copy
from ecb_doc_classes import RealInstance

def is_coref(string):
    return string != '-'

def is_not_coref(string):
    return string == '-'

def conll_file_name(is_gold, events, data_set):
    return 'ecb_plus_%s_%s.%s_conll'%('events' if events else 'all', data_set, 'key' if is_gold else 'response')

def convert_to_conll(tokens, conll_path, isgold, events_only=False, data_set='all', save=True):
    s = ''
    fname = 'ECB+/ecbplus_all'  # document.fname
    empty_line = fname+' -\n'

    coref_ids = {}
    continuing_mids = []
    for i, token in enumerate(tokens):
        if token.not_mention():
            s += empty_line
        else:
            s += fname + ' '
            put = False
            new_continuing_mids = []

            # putting starters and single-tokeners
            for mid in token.coref_ids:
                if mid not in continuing_mids:
                    if token.is_last_token_for_mid(mid): # single-token mention
                        s += '(' + str(mid) + ')'
                    else: # start new mention
                        s += '(' + str(mid)
                        new_continuing_mids.append(mid)
                    put = True
                    helpers.update_list_dict(coref_ids, mid, token)

            # putting enders
            cont_mid_list = list(continuing_mids)
            for mid in cont_mid_list:
                if token.is_last_token_for_mid(mid) or i == len(tokens)-1:
                    s += str(mid)+')'
                    continuing_mids.remove(mid)
                    put = True

            if not put:
                s += '-'

            s += '\n'
            continuing_mids += new_continuing_mids

    print '( = %d, ) = %d'%(s.count('('), s.count(')'))
    # assert s.count('(') == s.count(')')

    total_mentions = 0
    non_singleton_chains = 0
    singletons = 0
    for cid in coref_ids:
        total_mentions += len(coref_ids[cid])
        non_singleton_chains += 1 if len(coref_ids[cid])>1 else 0
        singletons += 1 if len(coref_ids[cid])==1 else 0

    print 'TOTAL MENTIONS:',total_mentions
    print 'TOTAL CHAINS:',non_singleton_chains
    print 'TOTAL SINGLETONS:',singletons

    if save:
        with open(conll_path + conll_file_name(isgold, events_only, data_set), 'w') as f:
            f.write('#begin document (ECB+/ecbplus_all); part 000\n')
            f.write(s)
            f.write('\n#end document\n')


class Documents(object):
    SAVE_FILE = 'parsed_ecb+.npy'

    def __init__(self, doc_list):
        self.d = {helpers.normalize_fname(doc.fname):doc for doc in doc_list}
        self.gold_mention_clusters = None
        self.sorted_docs = sorted(self.d.keys())

    def __getitem__(self, item):
        """
        :param item: Normalized filename string.
        :return: An ECBDocument or CoreDocument.
        """
        return self.d[item]

    def __iter__(self):
        for fname in self.sorted_docs:
            yield self.d[fname]

    def get_clusters(self):
        return self.gold_mention_clusters

    def keys(self):
        return self.d.keys()

    def convert(self, ecbdocs, clusters):
        """
        :type ecbdocs: Documents
        :type clusters: MentionCoreferenceClusters
        """
        for key in ecbdocs.keys():
            self[key].convert_tokens(ecbdocs[key], clusters.get_mentions_by_doc(ecbdocs[key].fname))
        self.gold_mention_clusters = clusters
        self.gold_mention_clusters.clean()

    def analyze(self, clean=False):
        if clean:
            self.gold_mention_clusters.clean()
        self.gold_mention_clusters.analyze()

    def convert_to_conll(self, conll_path, isgold, events_only=True, data_set='all'):
        convert_to_conll(self.get_all_tokens(), conll_path, isgold, events_only=events_only, data_set=data_set)

    def get_all_tokens(self, topics=helpers.ALL_TOPICS):
        tokens = []
        for document in self:
            if document.topic in topics:
                tokens += document.get_all_tokens()
        return tokens

    def filter_for_data_set(self, data_set):
        d = {'train':helpers.TRAIN, 'test':helpers.TEST, 'val':helpers.VAL}
        if data_set in d: # else return all
            to_remove = []
            for doc in self:
                if int(doc.topic) not in d[data_set]:
                    to_remove.append(doc)
            self.gold_mention_clusters.filter_for_data_set(d[data_set])
            for doc in to_remove:
                str_key = helpers.normalize_fname(doc.fname)
                del self.d[str_key]
                self.sorted_docs.remove(str_key)

    def get_doc_to_tokens_dict(self):
        return {fname:self.d[fname].get_all_tokens() for fname in self.sorted_docs}

    def serialize(self, save_file=SAVE_FILE, events_only=False):
        np.save(helpers.NPY_DATA_DIR+('events_' if events_only else 'all_')+save_file, np.array([self]))

    @classmethod
    def load(cls, load_file=SAVE_FILE, events_only=False):
        return np.load(helpers.NPY_DATA_DIR+('events_' if events_only else 'all_')+load_file)[0]


class MentionCoreferenceClusters(object):
    _PLACE_HOLDER = 'SINGLETON'

    def __init__(self):
        self.instances = []
        self.clusters = {}
        self.singleton_count = 1
        self.coref_ids = {}
        self.crt_coref_id = 1

    def add_mention(self, instance_id, mention, singleton=False):
        self.clusters[instance_id].append(mention)
        if MentionCoreferenceClusters.isactual(instance_id) or True:
            mention.set_coref_id(self.coref_ids[instance_id], singleton=singleton)

    def add_singleton_mention(self, mention):
        iid = self._PLACE_HOLDER + str(self.singleton_count)
        new_instance = RealInstance(mention.mclass + '_' + iid, None, 'NONE', iid, mention.fname)
        self.add_instance(new_instance)
        self.add_mention(iid, mention, singleton=True)
        self.singleton_count += 1

    def add_instance(self, real_instance):
        iid = real_instance.iid
        if iid not in self.clusters:
            self.instances.append(real_instance)
            self.clusters[iid] = []
            if MentionCoreferenceClusters.isactual(iid) or True:
                self.coref_ids[iid] = self.crt_coref_id
                self.crt_coref_id += 1

    def get_clusters(self, events_only=True, topics=helpers.ALL_TOPICS):
        clusters = {}
        for mention in self.itermentions():
            if mention.topic() in topics:
                if not events_only or mention.is_event():
                    helpers.update_list_dict(clusters, mention.coref_chain_id, mention)

    def get_mentions_by_doc(self, fname=None, topics=helpers.ALL_TOPICS):
        mentions_by_doc = {}
        mentions = []
        for mention in self.itermentions():
            if mention.topic() in topics:
                if fname and mention.fname == fname:
                    mentions.append(mention)
                else:
                    helpers.update_list_dict(mentions_by_doc, mention.fname, mention)
        if fname:
            return mentions
        return mentions_by_doc

    def get_mentions_by_class(self, topics=helpers.ALL_TOPICS):
        d = {}
        for mention in self.itermentions():
            if mention.topic() in topics:
                tag = mention.get_class()
                if tag.startswith('NEG'):
                    tag = 'ACTION'
                helpers.update_list_dict(d, tag.split('_')[0], mention)
        return d

    def itermentions(self):
        for iid in self.clusters:
            for mention in self.clusters[iid]:
                yield mention

    def clean(self):
        iids_to_clean = []
        for iid,lst in self.clusters.iteritems():
            if lst == []:
                iids_to_clean.append(iid)

        for instance in copy(self.instances):
            if instance.iid in iids_to_clean:
                self.clusters.pop(instance.iid, None)
                self.coref_ids.pop(instance.iid, None)
                self.instances.remove(instance)

        print 'NUMBER OF CLEANED CLUSTERS:',len(self.coref_ids)

    def analyze(self):
        unique_docs = set()
        tag_count_dict = self.get_mentions_by_class()
        for item in self.itermentions():
            unique_docs.add(item.fname)

        print '\nNumber of documents:',len(unique_docs)
        for k in tag_count_dict:
            print k, ':', len(tag_count_dict[k])
        print 'TOTAL:',sum([len(tag_count_dict[k]) for k in tag_count_dict])
        print 'TOTAL:',len([m for m in self.itermentions()]),'sanity check'

        for f in [lambda val: True]:
            l = [len(self.clusters[iid]) for iid in self.iteriids() if f(iid[:3])]
            # print '\nCoreference chain statistics:'
            print 'Number of instances:',len(self.get_instances())
            # print 'Number of CLUSTERS:',len(l)
            # print 'Average number of mentions to instance:', np.mean(l)
            # print 'Standard deviation of mentions to instance:', np.std(l)
            # print 'Max mentions to instance:', np.max(l)
            # print 'Percent of singletons instances:',float(len([v for v in l if v==1])) / len(l)
            # print '',

    def iterinstances(self, get_all=False):
        for inst in self.instances:
            if MentionCoreferenceClusters.isactual(inst.iid) or get_all:
                yield inst

    def iteriids(self):
        for inst in self.iterinstances():
            yield inst.iid

    def get_instances(self):
        return [inst for inst in self.iterinstances()]

    def filter_for_data_set(self, topics):
        instances_to_remove = []
        for instance in self.iterinstances(get_all=True):
            if instance.fname.startswith('9'):
                pass
            if int(helpers.get_topic(instance.fname)) not in topics:
                instances_to_remove.append(instance)
        for bad_inst in instances_to_remove:
            try:
                del self.clusters[bad_inst.iid]
            except KeyError:
                pass
            self.instances.remove(bad_inst)

    @classmethod
    def isactual(cls, iid):
        return not iid.startswith(cls._PLACE_HOLDER)
