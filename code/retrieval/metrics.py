import sys


def mean_average_precision(sort_data):
	# to do
	count_1 = 0
	sum_precision = 0
	for index in range(len(sort_data)):
		if sort_data[index][1] == 1:
			count_1 += 1
			sum_precision += 1.0 * count_1 / (index + 1)
	return sum_precision / count_1 if count_1 != 0 else 0.0


def mean_reciprocal_rank(sort_data):
	sort_lable = [s_d[1] for s_d in sort_data]
	assert 1 in sort_lable
	return 1.0 / (1 + sort_lable.index(1))


def precision_at_position_1(sort_data):
	if sort_data[0][1] == 1:
		return 1
	else:
		return 0


def recall_at_position_k_in_10(sort_data, k):
	sort_lable = [s_d[1] for s_d in sort_data]
	select_lable = sort_lable[:k]
	return 1.0 * select_lable.count(1) / sort_lable.count(1)


def evaluation_one_session(data):
	sort_data = sorted(data, key=lambda x: x[0], reverse=True)
	m_a_p = mean_average_precision(sort_data)
	m_r_r = mean_reciprocal_rank(sort_data)
	p_1 = precision_at_position_1(sort_data)
	r_1 = recall_at_position_k_in_10(sort_data, 1)
	r_2 = recall_at_position_k_in_10(sort_data, 2)
	r_5 = recall_at_position_k_in_10(sort_data, 5)
	return m_a_p, m_r_r, p_1, r_1, r_2, r_5


def evaluate(file_path, n=10):
	sum_m_a_p = 0
	sum_m_r_r = 0
	sum_p_1 = 0
	sum_r_1 = 0
	sum_r_2 = 0
	sum_r_5 = 0

	i = 0
	total_num = 0
	data = None
	with open(file_path, 'r') as infile:
		for line in infile:
			if i % n == 0:
				data = []

			tokens = line.strip().split('\t')
			if len(tokens) < 2:
				print('i', i, 'tokens', tokens)
			data.append((float(tokens[0]), int(tokens[1])))

			if i % n == n - 1:
				if 1 not in [s_d[1] for s_d in data]:
					continue
				total_num += 1
				m_a_p, m_r_r, p_1, r_1, r_2, r_5 = evaluation_one_session(data)
				sum_m_a_p += m_a_p
				sum_m_r_r += m_r_r
				sum_p_1 += p_1
				sum_r_1 += r_1
				sum_r_2 += r_2
				sum_r_5 += r_5

			i += 1

	print('total num: %s' % total_num)
	print('MAP: %s'.format(1.0*sum_m_a_p/total_num))
	print('MRR: {}'.format(1.0*sum_m_r_r/total_num))
	print('P@1: {}'.format(1.0*sum_p_1/total_num))
	print('R{}@1: {}'.format(n, str(1.0*sum_r_1/total_num)))
	print('R{}@2: {}'.format(n, str(1.0*sum_r_2/total_num)))
	print('R{}@5: {}'.format(n, str(1.0*sum_r_5/total_num)))
	# print('R10@1: %s' %(1.0*sum_r_1/total_num))
	# print('R10@2: %s' %(1.0*sum_r_2/total_num))
	# print('R10@5: %s' %(1.0*sum_r_5/total_num))
	return (1.0 * sum_m_a_p / total_num, 1.0 * sum_m_r_r / total_num, 1.0 * sum_p_1 / total_num,
	        1.0 * sum_r_1 / total_num, 1.0 * sum_r_2 / total_num, 1.0 * sum_r_5 / total_num)


if __name__ == '__main__':
	print(sys.argv)
	result = evaluate(sys.argv[1], sys.argv[2])
	# for r in result:
	# 	print(r)


"""
i = 0
cnt = 0
utterances, responses, labels = [], [], []
for line in open('data/test.txt'):
    contexts = line.strip().split('\t')
    uttes, resp, l = contexts[1:-1], contexts[-1], contexts[0]
    uttes = [utte.split() for utte in uttes]
    labels.append(int(l))
    if i % 10 == 9:
        if 1 in labels[i-9:]:
            cnt += 10
    i += 1
output: 
	i: 10000
	cnt: 6670
"""
