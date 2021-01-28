import os

text = open('original_data.txt', 'r').read()
string_list = text.split('\n\n')

i = 0
data_count = 50
max_data_length = 50
data_list = {}
phase_count = 3


for string in string_list:

    string = string.replace('\n', ' ')

    if len(string) > max_data_length:
        continue

    index = string.find(': ')
    if index == -1:
        continue

    key = string[0:index]

    if key in data_list:
        if len(data_list[key]) > (phase_count - 1):
            continue

    string = string[:index + 1] + string[index + 2:]

    if key not in data_list:
        data_list[key] = []

    data_list[key].append(string)

    if len(data_list[key]) == phase_count:
        i += 1

    if i > data_count - 1:
        break


writer_list = [open('rnn_' + str(p) + '.data', 'w') for p in range(phase_count)]

for key in data_list:
    string_list = data_list[key]

    if len(string_list) != phase_count:
        continue

    i = 0
    for string in string_list:
        writer_list[i].write(string + '\n')
        i += 1
