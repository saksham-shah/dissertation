from utils.load_batches import *

# Load batches: train_test

def test_train_test():
    test_arr = list(range(30))
    left, right = train_test(test_arr, 0, 10)
    assert len(left) == 27 and len(right) == 3
    assert right == [0, 1, 2] 
    assert left == list(range(3, 30))

    left, right = train_test(test_arr, 2, 6)
    assert len(left) == 25 and len(right) == 5
    assert right == [10, 11, 12, 13, 14] 
    assert left == list(range(10)) + list(range(15, 30))

    n = 4
    all_folds = []
    for i in range(n):
        left, right = train_test(test_arr, i, n)
        all_folds += right
    assert all_folds == test_arr

test_train_test()

print("All tests passed.")