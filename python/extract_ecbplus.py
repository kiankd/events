import csv
import simple_helpers as helpers
import xml.etree.ElementTree as ET
from ecb_doc_classes import EcbDocument
from general_classes import MentionCoreferenceClusters
from os import listdir

# Globals
SLASH_CHAR = '/'
BACK = '..' + SLASH_CHAR
ECB_PLUS_DIR = BACK + 'ECB+' + SLASH_CHAR
ECB_PLUS_DATA_FILES_PATH = ECB_PLUS_DIR + 'ECB+' + SLASH_CHAR
ECB_PLUS_ANNOTATED_SENTENCES_PATH = ECB_PLUS_DIR + 'ECBplus_coreference_sentences.csv'
ECB_PLUS_TOKENIZED_DIR = ECB_PLUS_DIR + 'Tokenized' + SLASH_CHAR
CORENLP_DIR = BACK + BACK + 'stanford-corenlp-full-2015-12-09' + SLASH_CHAR
CORENLP_ECB_PLUS_TOKENIZED = BACK + 'events_git' + SLASH_CHAR + ECB_PLUS_TOKENIZED_DIR.lstrip(BACK)

# Primary methods
def get_annotated_sentences(fname):
    f_to_sent_dict = {}
    with open(fname, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader.next()
        for row in csv_reader:
            fname = row[0] + '_' + row[1] + '.xml'
            sentence_num = int(row[2])
            helpers.update_list_dict(f_to_sent_dict, fname, sentence_num)
    return f_to_sent_dict

def get_ecb_data(topics_dirs, get_all=False, gold_count=False, clean=True, events_only=False):
    annotated_sentences_dict = get_annotated_sentences(ECB_PLUS_ANNOTATED_SENTENCES_PATH)

    mention_clusters = MentionCoreferenceClusters()
    documents = []
    print 'Processing ECB+ files from path: ', ECB_PLUS_DATA_FILES_PATH
    directories = listdir(ECB_PLUS_DATA_FILES_PATH)
    print 'There are %d topics.'%(len(directories))
    count = {}
    for directory in sorted(directories):
        topic = int(directory)
        if topic in topics_dirs:
            path = ECB_PLUS_DATA_FILES_PATH + directory + SLASH_CHAR
            for xml_file in sorted(listdir(path)):
                if get_all:
                    annotated_sentences_dict[xml_file] = None
                if get_all or xml_file in annotated_sentences_dict.keys():
                    documents.append(EcbDocument(xml_file,
                                                 path,
                                                 topic,
                                                 annotated_sentences_dict[xml_file],
                                                 mention_clusters,
                                                 get_all=get_all,
                                                 events_only=events_only))
                    if gold_count: count_ecb(count, path, xml_file, annotated_sentences_dict[xml_file])

    if gold_count: helpers.count_tag_vals(count)
    if clean: mention_clusters.clean()

    return documents, mention_clusters

def convert_docs_to_txt(documents):
    flist_name = CORENLP_DIR + 'ecbplus_filelist.txt'
    flist = []
    for ecbdoc in documents:
        newf = ECB_PLUS_TOKENIZED_DIR+helpers.extensionless(ecbdoc.fname)+'.txt'
        flist.append(CORENLP_ECB_PLUS_TOKENIZED + newf.lstrip(ECB_PLUS_TOKENIZED_DIR))
        with open(newf, 'w') as f:
            tokenized = ecbdoc.to_tokenized_file_string()
            if len(tokenized.split()) != len(ecbdoc.get_all_tokens()):
                print 'WARNING: DIFFERENT TOKENIZED LENGTHS! %d versus %d in file %s!'%(len(tokenized.split()), len(ecbdoc.get_all_tokens()), ecbdoc.fname)
            f.write(tokenized)

    with open(flist_name, 'w') as f:
        for filename in sorted(flist):
            f.write(filename+'\n')

def count_ecb(count_dict, path, xml_file, annotated_sentences):
    relevant_toks = []

    root = ET.parse(path + xml_file, parser=ET.XMLParser()).getroot()
    for tok in root.findall('token'):
        sentnum = int(tok.get('sentence'))
        if sentnum in annotated_sentences:
            relevant_toks.append(tok.get('t_id'))

    # get mentions
    for mention in root.find('Markables'):
        if mention.attrib.has_key('TAG_DESCRIPTOR'):
            continue
        mention_tok_ids = [child.get('t_id') for child in mention]
        if helpers.all_in_list(mention_tok_ids, relevant_toks):
            helpers.update_list_dict(count_dict, mention.tag, mention)

