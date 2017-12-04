
def get_svm_acc(y_pred_b, label_test):
	true_num = 0
	for i, pred in enumerate(y_pred_b):
		if label_test[i] == pred:
			true_num += 1
	return float(true_num)/len(y_pred_b)