# Test File Created By Haoxuan Zhu
# Last Modified Date: May 10, 2019
# Target: dot

import numpy as np
import transpy.numpy as tp

from transpy.scipy.sparse import csr_matrix  # as tp_csr_matrix
# from  scipy.sparse import csr_matrix

test_num = 10
corr_num = 0
scoreSheet = dict()
test_types = ["random", "sparse"]
for test_type in test_types:
    print("----------Testing "+test_type+"-------------")
    scoreSheet[test_type] = False
    corr_num = 0
    for i in range(test_num):
        try:
            if test_type == "random":
                a = np.random.randint(0,30,size=(500,500));
                b = np.random.randint(0,30,size=(500,500));
            if test_type == "eye":
                a = np.eye(i)
                b = np.eye(i, k=i)
            if test_type == "sparse":
                from scipy.sparse import random
                from scipy.stats import rv_continuous

                class CustomDistribution(rv_continuous):
                    def _rvs(self, *args, **kwargs):
                        return self._random_state.randn(*self._size)
                X = CustomDistribution(seed=2906)
                Y = X()  # get a frozen version of the distribution
                S1 = random(2, 2, density=0.87, random_state=2906, data_rvs=Y.rvs)
                S2 = random(2, 2, density=0.87, random_state=2906, data_rvs=Y.rvs)

                a = np.array(S1.A)
                b = np.array(S2.A)
                # a = csr_matrix(S1.A)
                # b = csr_matrix(S2.A)

            # print(a)
            # print(b)    
            dotnp = np.dot(a,b)
            dottp = tp.dot(a,b)
        except:
            print("Functionality Error: "+test_type)
            print(a)
            print(b)
            print(a.shape,b.shape)
            print("transpy gets: ")
            print(dotnp)
            print("numpy gets: ")
            print(dottp)
            break

        if np.allclose(dotnp,dottp):
        # if (dotnp==dottp).all():
            print(i,test_type+" correct")
            corr_num = corr_num + 1
        else:
            print("Calculation Error: "+test_type)
            print("Matrix a:\n ", a)
            print("Matrix b:\n ", b)
            print("transpy gets: ")
            print(dotnp)
            print("numpy gets: ")
            print(dottp)
            break
            
    if test_num == corr_num:
        scoreSheet[test_type]=True
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
# Report Generation
print("------FINAL REPORT-------")
for i in scoreSheet.keys():
    if scoreSheet[i] == True:
        print(test_num, i, " cases tested. All Pass! Congrats!")
    else:
        print(i, "test failed. Pls Check!!!")
