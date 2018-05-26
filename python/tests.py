from extract_ecbplus import get_ecb_data
from extract_corenlp import get_core_data
from general_classes import Documents
import simple_helpers as helpers

def get_all_data():
    topics = range(1,46)
    ecbs,clusters = get_ecb_data(topics)
    cores = get_core_data(topics)

    return Documents(ecbs),Documents(cores)

def token_equality_check(ecbs, cores):
    for key in ecbs.keys():
        doc = ecbs[key]
        coredoc = cores[key]
        assert len(doc.get_all_tokens()) == len(coredoc.get_all_tokens())

def test_coref_extraction(coredocs):
    singletons = 0
    corefs_dict = {}
    clust = coredocs.get_clusters()
    for mention in clust.itermentions():
        helpers.update_list_dict(corefs_dict, mention.coref_chain_id, mention)
        if mention.is_singleton:
            singletons += 1

    total_count = 0
    single_item_lists = 0
    for iid in corefs_dict:
        if len(corefs_dict[iid]) == 1:
            single_item_lists += 1
        total_count += len(corefs_dict[iid])

def find_bad_paranth():
    with open('../conll_corefs/ecb_plus_all.key_conll', 'r') as f:
        lines = f.readlines()
        s = ''
        for line in lines:
            s += line
        print s.count('(')
        print s.count(')')

        RANGE = 100

        # for c in s:


        # for line in lines:
        #     if line != '\n':
        #         value = line.split(' ')[1].strip('\n')
        #         if value != '-':
        #             if not value.startswith('(') and not value.endswith(')'):
        #                 print line

if __name__ == '__main__':
    # docs,cd = get_all_data()
    # token_equality_check(docs,cd)
    find_bad_paranth()

