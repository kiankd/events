from main import get_final_test_set
from collections import defaultdict

DOC_INCREMENT = 10000

def save_as_within_doc(file_name, gold_mentions):
	new_lines = []

	doc_names_to_new_coref_id = {}
	next_doc_increment = 0
	crt_mention_idx = 0
	
	with open(file_name, 'r') as f:	
		lines = f.readlines()
		for line in lines:
			if line.startswith('ECB+/ecbplus_all'):
				mention = gold_mentions[crt_mention_idx] # get mention

				splitten = line.split(' ')
				new_string = splitten[0] + ' '
				crt_coref_id = int(splitten[1].strip('(').strip(')\n')) # an int surrounded by parantheses, i.e. (100)

				# if we haven't hit this doc yet, make new id-base for it
				if not mention.fname in doc_names_to_new_coref_id:
					next_doc_increment += DOC_INCREMENT
					doc_names_to_new_coref_id[mention.fname] = next_doc_increment

				# update id to make only within doc
				crt_coref_id += doc_names_to_new_coref_id[mention.fname]

				# add the modified line
				new_lines.append(new_string + '(%d)'%crt_coref_id)
				crt_mention_idx += 1
			else:
				new_lines.append(line.strip('\n'))

	splitten = file_name.split('.')
	new_file_name = splitten[0] + '_WITHINDOC_.' + splitten[1]
	write_file(new_file_name)

def save_as_cross_doc(file_name, gold_mentions, idx_to_new_idx, new_idx_to_chain):
	crt_mention_idx = 0
	predictions = []
	with open(file_name, 'r') as f:	
		lines = f.readlines()
		for line in lines:
			if line.startswith('ECB+/ecbplus_all'):
				mention = gold_mentions[crt_mention_idx] # get mention
				splitten = line.split(' ')
				new_string = splitten[0] + ' '
				crt_coref_id = int(splitten[1].strip('(').strip(')\n')) # an int surrounded by parantheses, i.e. (100)
				predictions.append(crt_coref_id)
				crt_mention_idx += 1

	# now we have the list of predictions
	# lets find the mention-changed that have to be merged
	chain_ids_to_merge = defaultdict(list)
	for i, coref_pred in enumerate(predictions):
		chain_ids_to_merge[idx_to_new_idx[i]].append(coref_pred)

	new_chain_num = 1
	for lst in chain_ids_to_merge.values():
		lst.append(new_chain_num)
		new_chain_num += 1

	# now make the new predictions based on that merge
	new_chains = []
	crt_doc = gold_mentions[0].fname
	doc_was_put = set([])

	for i, coref_pred in enumerate(predictions):
		doc = gold_mentions[i].fname
		if crt_doc != doc:
			crt_doc = doc
			doc_was_put.clear()

		new_i = idx_to_new_idx[i]
		chain_mergers = chain_ids_to_merge[new_i]
		
		tup = (doc, chain_mergers[-1],)
		if tup in doc_was_put:
			continue

		new_chains.append(tup[1])
		doc_was_put.add(tup)

	# writing new file
	new_lines = ['#begin document (ECB+/ecbplus_all); part 000']
	for chain in new_chains:
		new_lines.append('ECB+/ecbplus_all (%d)'%chain)
	new_lines.append('')
	new_lines.append('#end document')
	new_lines.append('')

	for l in new_lines[:40]:
		print l

	# write the new file
	splitten = file_name.split('.')
	new_file_name = splitten[0] + '_CROSSDOC_.' + splitten[1]
	# write_file(new_file_name)

def gold_to_pure_cdec(gold_mentions):
	idx_to_new_idx = {}
	new_idx_to_chain = {}
	doc_level_coref_idxs = {}

	crt_doc = gold_mentions[0].fname
	crt_new_idx = 0
	
	for i,mention in enumerate(gold_mentions):
		chain_id = mention.coref_chain_id

		if mention.fname != crt_doc:
			crt_doc = mention.fname
			doc_level_coref_idxs = {}

		if not chain_id in doc_level_coref_idxs:
			doc_level_coref_idxs[chain_id] = crt_new_idx
			idx_to_new_idx[i] = crt_new_idx
			new_idx_to_chain[idx_to_new_idx[i]] = chain_id
			crt_new_idx += 1

		# else, there is within-doc coref, so we should remove it by merging them
		else: 
			idx_to_new_idx[i] = doc_level_coref_idxs[chain_id] # set to same index

	return idx_to_new_idx, new_idx_to_chain


def write_file(file_name):
	with open(file_name, 'w') as f:
		for line in new_lines:
			f.write(line+'\n')


if __name__ == '__main__':
	_, _, test_mentions = get_final_test_set(events_only=True)

	idx_to_new_idx, new_idx_to_chain = gold_to_pure_cdec(test_mentions)

	for i in xrange(40):##xrange(len(test_mentions)):
		print i, test_mentions[i].coref_chain_id, idx_to_new_idx[i], new_idx_to_chain[idx_to_new_idx[i]], test_mentions[i].fname

	files_to_modify = [
	]

	for fname in files_to_modify[:1]:
		print('Running file %s'%fname)
		# save_as_within_doc(fname, test_mentions)
		save_as_cross_doc(fname, test_mentions, idx_to_new_idx, new_idx_to_chain)
