titleFlag = {}

def print_list(title, table_list):

	titleKey = make_title_key(table_list)

	if titleKey not in titleFlag:
		titleFlag[titleKey] = True
		show_title_list(title)

	show_data_list(table_list)


def show_title_list(title):

	print('')
	print('='*70)

	print(title)

	print('-'*70)


def show_title(table):

	template = ''
	for key in table:
		template += '{' + key + ':30}'

	print('')
	print('='*70)
	colmun = {}

	for key in table:
		colmun[key] = key

	print(template.format(**colmun))

	print('-'*70)


def show_data_list(table_list):

	for key in table_list:
		dict = {}
		data = str(table_list[key])
		template = '{key:30}{data:30}'
		dict['key'] = key
		dict['data'] = data

		print(template.format(**dict))

	print('-'*70)


def show_data(table):

	template = ''

	for key in table:
		template += '{' + key + ':30}'

	firstKey = list(table.keys())[0]
	length = len(table[firstKey])

	for i in range(length):
		dict = {}
		for key in table:
			dict[key] = str(table[key][i])

		print(template.format(**dict))


def make_title_key(table):

	key_list = sorted(list(table.keys()))
	return ''.join(key_list)


def print_table(table):

	titleKey = make_title_key(table)

	if titleKey not in titleFlag:
		titleFlag[titleKey] = True
		show_title(table)

	show_data(table)
