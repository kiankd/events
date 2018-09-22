#!/bin/bash

git clone https://github.com/conll/reference-coreference-scorers.git 
curl -O kyoto.let.vu.nl/repo/ECB+_LREC2014.zip
unzip ECB+_LREC2014.zip
rm ECB+_LREC2014.zip
mv ECB+_LREC2014 ECB+
mkdir ECB+/ECB+/
unzip ECB+/ECB+.zip -d ECB+/
rm ECB+/ECB+.zip
unzip Parsed.zip -d ECB+/
rm Parsed.zip
tar -xzf vocab_files.tar.gz
rm vocab_files.tar.gz
