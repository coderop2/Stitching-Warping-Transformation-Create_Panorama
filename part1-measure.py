import cv2
import numpy as np
import sys
def get_tp_count(l):
	tp = 0
	n = len(l)
	for i in range(n):
		for j in range(i+1,n):
			if (l[i].split('_')[0] == l[j].split('_')[0]):
				tp = tp + 1
	return tp
def get_tn_count(l1, l2):
	tn = 0
	n1 = len(l1)
	n2 = len(l2)
	for i in range(n1):
		for j in range(n2):
			if (l1[i].split('_')[0] != l2[j].split('_')[0]):
				tn = tn + 1
	return tn
def get_metrics(label_dict):
	values = []
	n = 0
	for _, value in label_dict.items():
		n = n + len(value)
		values.append(value)
	k = len(values)
	tp = 0
	tn = 0
	for i in range(k):
		tp = tp + get_tp_count(values[i])
		for j in range(i+1, k):
			tn = tn + get_tn_count(values[i], values[j])
	f = open('measurements.txt', 'w')
	f.write('TP:'+str(tp)+'\n')
	f.write('TN:'+str(tn)+'\n')
	f.write('PairWise Accuracy:'+str((tp+tn)*2 / (n*(n-1))))
if __name__=="__main__":
	output_file = sys.argv[1]
	i = 0
	label_dict = {}
	with open(output_file,'r') as f:
		lines = [line.rstrip() for line in f]
		for line in lines:
			label_dict[i] = line.split()
			i = i + 1
	get_metrics(label_dict)



