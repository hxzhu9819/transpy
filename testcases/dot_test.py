import numpy as np
import transpy.numpy as tp
from transpy.scipy.sparse import csr_matrix  # as tp_csr_matrix

FIXED_SAMPLE = np.array([[0 ,0 ,1 ,1 ,1 ,0 ,1 ,0]
                    , [1, 1, 0, 0, 0, 0, 1, 1]
                    , [1, 0, 0, 0, 1, 0, 1, 1]
                    , [1, 1, 1, 1, 1, 0, 0, 0]
                    , [0, 0, 0, 1, 1, 1, 1, 1]
                    , [1, 0, 0, 0, 1, 0, 1, 0]
                    , [0, 1, 1, 0, 1, 0, 0, 0]
                    , [0, 0, 1, 1, 0, 1, 0, 1]])

test_num = 10
corr_num = 0
for i in range(test_num):
    try:
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
        dottp = csr_matrix(a).dot(csr_matrix(b))
        dotnp = np.matmul(a, b)

    except:
        print("Error:")
        print(a)
        print(b)
        print("transpy gets: ")
        print(dottp)
        print("numpy gets: ")
        print(dotnp)
        exit(-1)

    if (dotnp==dottp).all():
        print(i," correct")
        corr_num = corr_num + 1
    else:
        print("Error:")
        print("Matrix a: ", a)
        print("Matrix b: ", b)
        print("transpy gets: ")
        print(dottp)
        print("numpy gets: ")
        print(dotnp)
        exit(-1)

# print("transpy gets: ")
# print(dotnp)
# print("numpy gets: ")
# print(dottp)

if test_num == corr_num:
    print(test_num,"randomly-created cases tested. All Pass! Congrats!")



"""
print("-----------EDGE CASE TESTS START--------------")
corr_num_eye = 0
for i in range(test_num):
    try:
        a = np.eye(i)
        b = np.eye(i,k=i)
        dotnp = np.dot(a,b)
        dottp = tp.dot(a,b)
    except:
        print("Error: eye")
        print(a)
        print(b)
        exit(-1)
    if (dotnp==dottp).all():
        print(i," correct")
        corr_num_eye = corr_num_eye + 1
    else:
        print("Error:")
        print("Matrix a: ", a)
        print("Matrix b: ", b)
        print("transpy gets: ")
        print(dotnp)
        print("numpy gets: ")
        print(dottp)
        exit(-1)
"""



from scipy.sparse import random
from scipy.stats import rv_continuous
class CustomDistribution(rv_continuous):
    def _rvs(self, *args, **kwargs):
        return self._random_state.randn(*self._size)
X = CustomDistribution(seed=2906)
Y = X()  # get a frozen version of the distribution
S1 = random(3, 4, density=0.25, random_state=2906, data_rvs=Y.rvs)
S2 = random(3, 4, density=0.25, random_state=2906, data_rvs=Y.rvs)
a = np.array(S1.A)
b = np.array(S2.A)












# Report Generation
if test_num == corr_num:
    print(test_num,"randomly-created cases tested. All Pass! Congrats!")
# if test_num == corr_num_eye:
#    print(test_num,"eye cases tested. All Pass! Congrats!")
