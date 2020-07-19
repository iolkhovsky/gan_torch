

def accuracy(pred_vector, target_vector, norm=True):
    total = 1
    for dim in pred_vector.shape:
        total = total * dim
    correct_cnt = pred_vector.eq(target_vector.view_as(pred_vector)).sum().item()
    if norm:
        correct_cnt /= total
    return correct_cnt
