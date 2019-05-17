import transpy.numpy as tp
import numpy as np
from transpy.scipy.sparse import csr_matrix #as tp_csr_matrix
#from  scipy.sparse import csr_matrix

FIXED_SAMPLE = np.array([[0 ,0 ,1 ,1 ,1 ,0 ,1 ,0]
                    , [1, 1, 0, 0, 0, 0, 1, 1]
                    , [1, 0, 0, 0, 1, 0, 1, 1]
                    , [1, 1, 1, 1, 1, 0, 0, 0]
                    , [0, 0, 0, 1, 1, 1, 1, 1]
                    , [1, 0, 0, 0, 1, 0, 1, 0]
                    , [0, 1, 1, 0, 1, 0, 0, 0]
                    , [0, 0, 1, 1, 0, 1, 0, 1]])

ver = False
test_num = 5
corr_num = 0
scoreSheet = dict()
# test_types = ["sparse-normal", "sparse-sparse", "sparse-dense", "dense-normal","dense-sparse","dense-dense"]
test_types = ["dense-sparse"]
for test_type in test_types:
    print("----------Testing " + test_type + "-------------")
    scoreSheet[test_type] = False
    corr_num = 0

    for i in range(test_num):
        # Generate matrices A and B
        l = np.random.randint(1, 100)
        w = np.random.randint(1, 100)
        val_1 = np.random.randint(1, 100)
        val_2 = np.random.randint(2, 100)
        # Construct randomized A, B
        a = np.random.randint(0, val_1, size=(l, w))
        b = np.random.randint(1, val_2, size=(w, l))
        if i % 2 == 0:
            a = FIXED_SAMPLE
            b = FIXED_SAMPLE
        try:
            if test_type == "sparse-normal":
                A = csr_matrix(a)
                B = b
            if test_type == "sparse-sparse":
                A = csr_matrix(a)
                B = csr_matrix(b)
            if test_type == "sparse-dense":
                A = csr_matrix(a)
                B = csr_matrix(b).todense()
            if test_type == "dense-normal":
                A = csr_matrix(a).todense()
                B = b
            if test_type == "dense-sparse":
                A = csr_matrix(a).todense()
                B = csr_matrix(b)
            if test_type == "dense-dense":
                A = csr_matrix(a).todense()
                B = csr_matrix(b).todense()
            print(type(a),type(b))
            dotnp = np.matmul(a, b)
            dottp = A.dot(B)

            if ver:
                print("Matrices:")
                print(a)
                print(b)
                print("np:\n", dotnp)
                print("tp:\n", dotnp)

        except:
            print("Functionality Error: " + test_type)
            print(a)
            print(b)
            print(a.shape, b.shape)
            print("transpy gets: ")
            print(dottp)
            print("numpy gets: ")
            print(dotnp)
            errora = a
            errorb = b
            break



        if np.allclose(tp.dot(a,b),np.matmul(a,b)):
        # if np.allclose(dotnp,dottp):
        # if (dotnp == dottp).all():
            print(i, test_type + " correct")
            corr_num = corr_num + 1


    if test_num == corr_num:
        scoreSheet[test_type] = True
        print(test_num, "randomly-created cases tested. All Pass! Congrats!")


# Report Generation
print("------FINAL REPORT-------")
print("Testing multidot")
for i in scoreSheet.keys():
    if scoreSheet[i] == True:
        print(test_num, i, "cases tested. All Pass! Congrats!")
    else:
        print(i, "test failed. Pls Check!!!")
        print("Try:")
        print(errora)
        print(errorb)
