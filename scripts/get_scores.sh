#!/usr/bin/env bash

echo "Doing both WITHIN and CROSS document evaluation."
echo "If you want to do WITHIN only, edit the line defining the variable KEY such that: "
echo "    KEY=ecb_plus_events_test_mention_based_WITHIN_.key_conll"
echo "    (and then make sure that your .response_conll is also a response for _within_ doc coreference!)"


CONLLPATH=../results/
KEY=${CONLLPATH}ecb_plus_events_test_mention_based.key_conll
RESPONSE=${CONLLPATH}ecb_plus.response_conll

if [ $# -eq 1 ]
    then
        RESPONSE=${CONLLPATH}${1}
fi

if [ $# -eq 2 ]
    then
        KEY=${CONLLPATH}${1}
        RESPONSE=${CONLLPATH}${2}
fi

echo "Key file: "${KEY}
echo "Response file: "${RESPONSE}

../reference-coreference-scorers/scorer.pl all ${KEY} ${RESPONSE} none
