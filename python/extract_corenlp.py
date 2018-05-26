from os import listdir
from corenlp_classes import CoreDocument
import simple_helpers as helpers


CORENLP_DATA_DIR = '../ECB+/Parsed/'

def get_core_data(topics_list):
    docs = []
    print 'Processing CoreNLP files from path: ',CORENLP_DATA_DIR
    for xml_file in sorted(listdir(CORENLP_DATA_DIR)):
        if int(helpers.get_topic(xml_file)) in topics_list:
            docs.append(CoreDocument(CORENLP_DATA_DIR, xml_file))
    return docs
