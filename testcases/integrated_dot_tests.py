import transpy.numpy as tp
import numpy as np
from transpy.scipy.sparse import csr_matrix
import time
# from scipy.sparse import csr_matrix

#from  scipy.sparse import csr_matrix
np.set_printoptions(threshold=np.inf)

v = False
test_num = 10

FIXED_SAMPLE = np.array([[0, 0, 1, 1, 1, 0, 1, 0]
                    , [1, 1, 0, 0, 0, 0, 1, 1]
                    , [1, 0, 0, 0, 1, 0, 1, 1]
                    , [1, 1, 1, 1, 1, 0, 0, 0]
                    , [0, 0, 0, 1, 1, 1, 1, 1]
                    , [1, 0, 0, 0, 1, 0, 1, 0]
                    , [0, 1, 1, 0, 1, 0, 0, 0]
                    , [0, 0, 1, 1, 0, 1, 0, 1]])


def check_dot(l, r, oa, ob, v=False):
    # convert input to np array
    # l = np.asarray(l.astype(float))
    # r = np.asarray(r.astype(float))
    if v:
        print("Input Matrices:")
        print(type(l), l)
        print(type(r), r)

    try:
        err = False
        # dottp = tp.dot(l, r)
        # dotnp = np.dot(l, r)
        dottp = l.dot(r)

        dotnp = np.matmul(oa, ob)

        if not (dottp == dotnp).all():
            print('Potential Calculation error: ')
            print('Matrix A')
            print(l)
            print('Matrix B')
            print(r)
            print('Transpy gets')
            print(dottp)
            print('NumPy gets')
            print(dotnp)
            # exit(-1)
            return 0
        # if v is True, then print detailed test cases
        if v:
            print("Matrices:")
            print(l)
            print(r)
            print("np:\n", dotnp)
            print("tp:\n", dotnp)

    except MemoryError as error:
        time.sleep(1)
        print('Mem Error:', error)
        print('Matrix A.size:', type(l), l.shape)
        print('Matrix B.size:', type(r), r.shape)
        err = True
        # exit(-1)
        return 0, err

    except Exception as exception:
        time.sleep(1)
        print('Error Type:', exception)
        print('Matrix B.size:', type(l), r.shape)
        print('Matrix A.size:', type(r), l.shape)
        err = True
        # exit(-1)
        return 0, err

    return 1,err


def generate_random_array(size_max=100, val_max=100, vec=False):

    if vec:
        c = 1
    else:
        c = np.random.randint(1, size_max)
    r = np.random.randint(1, size_max)
    val_1 = np.random.randint(1, val_max)
    val_2 = np.random.randint(2, val_max)
    # Construct randomized A, B
    mata = np.random.randint(0, val_1, size=(r, c))
    matb = np.random.randint(1, val_2, size=(c, r))
    if np.random.randint(1, 10) % 2 == 0:
        mata = FIXED_SAMPLE
        matb = FIXED_SAMPLE
    return mata, matb


corr_num = 0
scoreSheet = dict()

print('------Matrix-Matrix dot-------')
test_types = ["mm-sparse-normal", "mm-sparse-sparse", "mm-sparse-dense", "mm-dense-normal","mm-dense-sparse","mm-dense-dense"]
for test_type in test_types:

    print("----------Testing " + test_type + "-------------")
    scoreSheet[test_type] = False
    corr_num = 0

    for i in range(test_num):
        # Generate matrices A and B
        a, b = generate_random_array(size_max=100, val_max=100, vec=False)
        oa = a
        ob = b
        try:
            if test_type == "mm-sparse-normal":
                a = csr_matrix(a)
                b = b
            if test_type == "mm-sparse-sparse":
                a = csr_matrix(a)
                b = csr_matrix(b)
            if test_type == "mm-sparse-dense":
                a = csr_matrix(a)
                b = csr_matrix(b).todense()
            if test_type == "mm-dense-normal":
                a = csr_matrix(a).todense()
                b = b
            if test_type == "mm-dense-sparse":
                a = csr_matrix(a).todense()
                b = csr_matrix(b)
            if test_type == "mm-dense-dense":
                a = csr_matrix(a).todense()
                b = csr_matrix(b).todense()

        except:
            print('check csr_matrix!')
            exit(-1)
        res, fun_err = check_dot(a, b, oa, ob, v)
        if res:
            print(i, test_type + " correct")
            corr_num = corr_num + 1

        if fun_err:
            break

    if test_num == corr_num:
        scoreSheet[test_type] = True
        print(test_num, "randomly-created cases tested. All Pass! Congrats!")


print('------Vector-Vector dot-------')
test_types = ["vv-sparse-normal", "vv-sparse-sparse", "vv-sparse-dense", "vv-dense-normal","vv-dense-sparse","vv-dense-dense"]
for test_type in test_types:

    print("----------Testing " + test_type + "-------------")
    scoreSheet[test_type] = False
    corr_num = 0

    for i in range(test_num):
        # Generate matrices A and B
        a, b = generate_random_array(size_max=100,val_max=100,vec=True)
        oa = a
        ob = b
        try:
            if test_type == "vv-sparse-normal":
                a = csr_matrix(a)
                b = b
            if test_type == "vv-sparse-sparse":
                a = csr_matrix(a)
                b = csr_matrix(b)
            if test_type == "vv-sparse-dense":
                a = csr_matrix(a)
                b = csr_matrix(b).todense()
            if test_type == "vv-dense-normal":
                a = csr_matrix(a).todense()
                b = b
            if test_type == "vv-dense-sparse":
                a = csr_matrix(a).todense()
                b = csr_matrix(b)
            if test_type == "vv-dense-dense":
                a = csr_matrix(a).todense()
                b = csr_matrix(b).todense()
        except:
            print('check csr_matrix!')
            exit(-1)

        res, fun_err = check_dot(a, b, oa, ob, v)
        if res:
            print(i, test_type + " correct")
            corr_num = corr_num + 1

        if fun_err:
            break

    if test_num == corr_num:
        scoreSheet[test_type] = True
        print(test_num, "randomly-created cases tested. All Pass! Congrats!")


# Report Generation
print("------FINAL REPORT-------")
print("Testing multidot")
for i in scoreSheet.keys():
    if scoreSheet[i]:
        print(test_num, i, "cases tested. All Pass! Congrats!")
    else:
        print(i, "test failed. Pls Check!!!")
