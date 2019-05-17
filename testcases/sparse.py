import transpy.numpy as np
# import numpy as np
from transpy.scipy.sparse import csr_matrix  # as tp_csr_matrix
# from  scipy.sparse import csr_matrix

import numpy as std_np
import time

std_np.set_printoptions(threshold=std_np.inf)
FIXED_SAMPLE = np.array([[0, 0, 1, 1, 1, 0, 1, 0]
                            , [1, 1, 0, 0, 0, 0, 1, 1]
                            , [1, 0, 0, 0, 1, 0, 1, 1]
                            , [1, 1, 1, 1, 1, 0, 0, 0]
                            , [0, 0, 0, 1, 1, 1, 1, 1]
                            , [1, 0, 0, 0, 1, 0, 1, 0]
                            , [0, 1, 1, 0, 1, 0, 0, 0]
                            , [0, 0, 1, 1, 0, 1, 0, 1]])

# Size of the random tester
FUZZER_SIZE = 5


def compare_mtx(left, right):
    # Enable the following line when trying to expose TM runtime memory errors bypassing the correctness check.
    #################
    # return        #
    #################
    left = std_np.asarray(left.astype(float))
    right = std_np.asarray(right.astype(float))
    try:
        # if (left == right).all() == False:
        if std_np.allclose(left, right) == False:
            print('Found differences in the two matrices')
            print(left)
            print(right)
            exit(0)
    except MemoryError as error:
        print('Error Type:', error)
        print('Left size:', left.shape)
        print('Right size:', right.shape)
        exit(0)

    except Exception as exception:
        print('Error Type:', exception)
        print('Left size:', left.shape)
        print('Right size:', right.shape)
        exit(0)


print('part 1')
print('test_element wise + - * \\ exp power')
for i in range(FUZZER_SIZE):
    l = std_np.random.randint(1, 10)
    w = std_np.random.randint(1, 10)
    val_1 = std_np.random.randint(1, 10)
    val_2 = std_np.random.randint(2, 10)
    # Construct randomized A, B

    A = np.random.randint(0, val_1, size=(l, w))
    B = np.random.randint(1, val_2, size=(l, w))
    # Compare transpy result with standard numpy result
    compare_mtx(np.multiply(A, B), std_np.multiply(A, B))
    compare_mtx(np.divide(A, B), std_np.divide(A, B))
    compare_mtx(np.power(A, B), std_np.power(A, B))

    # The following two tests may cause std::bad_alloc in tm.config() when testing size is large.
    #compare_mtx(np.exp(A), std_np.exp(A))
    #compare_mtx(np.exp(B), std_np.exp(B))

    A = FIXED_SAMPLE
    B = FIXED_SAMPLE
    compare_mtx(np.multiply(A, B), std_np.multiply(A, B))
    compare_mtx(np.power(A, B), std_np.power(A, B))

print('#####       part 1 passed     #######')
time.sleep(2)

print('part 2')
print('test_inner / outer / dot product')
for i in range(FUZZER_SIZE):
    l = std_np.random.randint(1, 100)
    w = std_np.random.randint(1, 100)
    val_1 = std_np.random.randint(1, 100)
    val_2 = std_np.random.randint(2, 100)
    # Construct randomized A, B
    A = np.random.randint(0, val_1, size=(l, w))
    B = np.random.randint(1, val_2, size=(w, l))
    # Compare transpy result with standard numpy result
    # compare_mtx(np.inner(A, B.T), std_np.inner(A, B.T))
    compare_mtx(np.outer(A, B.T), std_np.outer(A, B.T))
    # compare_mtx(A.dot(B), std_np.dot(A, B))

    A = FIXED_SAMPLE
    B = FIXED_SAMPLE
    # compare_mtx(np.inner(A, B.T), std_np.inner(A, B.T))
    compare_mtx(np.outer(A, B.T), std_np.outer(A, B.T))
    # compare_mtx(A.dot(B), std_np.dot(A, B))
print('#####       part 2 passed     #######')
time.sleep(2)

print('part 3')
print('test_dot product of csr matrix & dense matrix')
for i in range(FUZZER_SIZE):
    l = std_np.random.randint(1, 100)
    w = std_np.random.randint(1, 100)
    val_1 = std_np.random.randint(1, 100)
    val_2 = std_np.random.randint(2, 100)
    # Construct randomized A, B
    A = np.random.randint(0, val_1, size=(l, w))
    B = np.random.randint(1, val_2, size=(w, l))
    if i % 2 == 0:
        A = FIXED_SAMPLE
        B = FIXED_SAMPLE


    # sparse.normal
    compare_mtx(csr_matrix(A).dot(B), std_np.matmul(A, B))

    # sparse.sparse will cause error
    # compare_mtx(np.dot(csr_matrix(A),csr_matrix(B)), std_np.matmul(A, B))
    csr_A = csr_matrix(A)
    csr_B = csr_matrix(B)
    # compare_mtx(csr_A.dot(csr_B), std_np.matmul(A, B))

    # result = np.dot(csr_A, csr_B)
    # compare_mtx(result, std_np.matmul(A, B))

    # sparse.dense
    compare_mtx(csr_matrix(A).dot(csr_matrix(B).todense()), std_np.matmul(A, B))

    # dense.normal
    compare_mtx(csr_matrix(A).todense().dot(B), std_np.matmul(A, B))

    # dense.sparse will cause error
    # compare_mtx(csr_matrix(A).todense().dot(csr_matrix(B)), std_np.matmul(A, B))

    # dense.dense
    compare_mtx(csr_matrix(A).todense().dot(csr_matrix(B).todense()), std_np.matmul(A, B))

print('#####       part 3 passed     #######')
time.sleep(2)

print('part 4')
# vector vector opration
for i in range(FUZZER_SIZE):
    l = std_np.random.randint(1, 100)
    w = 1
    val_1 = std_np.random.randint(1, 100)
    val_2 = std_np.random.randint(2, 100)
    # Construct randomized A, B vector
    A = np.random.randint(0, val_1, size=(l, w))
    B = np.random.randint(1, val_2, size=(w, l))
    if i % 2 == 0:
        A = FIXED_SAMPLE
        B = FIXED_SAMPLE

    compare_mtx(A.dot(B), std_np.matmul(A, B))
    compare_mtx(np.inner(A, B.T), std_np.inner(A, B.T))
    compare_mtx(np.outer(A, B.T), std_np.outer(A, B.T))

    # The following two tests may cause std::bad_alloc in tm.config() when testing size is large.
    # compare_mtx(np.exp(A), std_np.exp(A))
    # compare_mtx(np.exp(B), std_np.exp(B))

    # normal.normal
    compare_mtx(A.dot(B), std_np.matmul(A, B))

    # sparse.normal
    compare_mtx(csr_matrix(A).dot(B), std_np.matmul(A, B))

    # sparse.sparse will cause error
    # compare_mtx(np.dot(csr_matrix(A),csr_matrix(B)), std_np.matmul(A, B))
    csr_A = csr_matrix(A)
    csr_B = csr_matrix(B)
    # compare_mtx(csr_A.dot(csr_B), std_np.matmul(A, B))

    # result = np.dot(csr_A, csr_B)
    # compare_mtx(result, std_np.matmul(A, B))

    # sparse.dense
    compare_mtx(csr_matrix(A).dot(csr_matrix(B).todense()), std_np.matmul(A, B))

    # dense.normal
    compare_mtx(csr_matrix(A).todense().dot(B), std_np.matmul(A, B))

    # dense.sparse will cause error
    # compare_mtx(csr_matrix(A).todense().dot(csr_matrix(B)), std_np.matmul(A, B))

    # dense.dense
    compare_mtx(csr_matrix(A).todense().dot(csr_matrix(B).todense()), std_np.matmul(A, B))

print('#####       part 4 passed     #######')