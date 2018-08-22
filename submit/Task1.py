#encoding:utf-8
"""
下面的文件将会从csv文件中读取读取短信与电话记录，
你将在以后的课程中了解更多有关读取文件的知识。
"""
import csv
with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)


"""
任务1：
短信和通话记录中一共有多少电话号码？每个号码只统计一次。
输出信息：
"There are <count> different telephone numbers in the records."
"""

phone_numbers = []
for i in texts:
    incomming_number = i[0]
    answer_number = i[1]
    if incomming_number not in phone_numbers:
        phone_numbers.append(incomming_number)
    if answer_number not in phone_numbers:
        phone_numbers.append(answer_number)

for i in calls:
    incomming_number = i[0]
    answer_number = i[1]
    if incomming_number not in phone_numbers:
        phone_numbers.append(incomming_number)
    if answer_number not in phone_numbers:
        phone_numbers.append(answer_number)
    
print("There are %d different telephone numbers in the records." % len(phone_numbers))
