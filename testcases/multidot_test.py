# Test File Created By Haoxuan Zhu
# Last Modified Date: May 11, 2019
# Target: inner

import numpy as np
import transpy.numpy as tp

ver = False
test_num = 10
corr_num = 0
scoreSheet = dict()
test_types = ["random", "sparse", "eye"]
for test_type in test_types:
    print("----------Testing " + test_type + "-------------")
    scoreSheet[test_type] = False
    corr_num = 0
    for i in range(test_num):
        try:
            if test_type == "special":
                a = np.arange(24).reshape((2, 3, 4))
                b = np.arange(4)
            if test_type == "complex":
                a = np.array([[np.random.randint(0, 5)*i*0.123124312545j+np.random.randint(0, 5)*0.512213124314312545j, np.random.randint(0, 5)-np.random.randint(0, 5)*2.3j], [0, 1.2j]])
                b = np.array([[np.random.randint(0, 5) * i * 0.332 + np.random.randint(0, 10) * 0.003j, 1 + 2j],
                              [2 + np.random.randint(0, 5) * 3j, 2 + np.random.randint(0, 5) * 0.21142j]])
                c = np.array([[np.random.randint(0, 5) * i * 0.332 + np.random.randint(0, 10) * 0.003j, 1 + 2j],
                              [2 + np.random.randint(0, 5) * 3j, 2 + np.random.randint(0, 5) * 0.21142j]])
                d = np.array([[np.random.randint(0, 5) * i * 0.332 + np.random.randint(0, 10) * 0.003j, 1 + 2j],
                              [2 + np.random.randint(0, 5) * 3j, 2 + np.random.randint(0, 5) * 0.21142j]])
            if test_type == "random":
                a = np.random.random((10000, 100))
                b = np.random.random((100, 1000))
                c = np.random.random((1000, 5))
                d = np.random.random((5, 333))
            if test_type == "eye":
                a = np.eye(i)
                b = np.eye(i, k=i)
                c = np.eye(i,k=i+1)
                d = np.eye(i,k=i+2)
            if test_type == "sparse":
                from scipy.sparse import random
                from scipy.stats import rv_continuous


                class CustomDistribution(rv_continuous):
                    def _rvs(self, *args, **kwargs):
                        return self._random_state.randn(*self._size)


                X = CustomDistribution(seed=2906)
                Y = X()  # get a frozen version of the distribution
                S1 = random(1, i, density=0.87, random_state=2906, data_rvs=Y.rvs)
                S2 = random(i, 4, density=0.87, random_state=2906, data_rvs=Y.rvs)
                S3 = random(4, 100, density=0.87, random_state=2906, data_rvs=Y.rvs)
                S4 = random(100, 2000, density=0.87, random_state=2906, data_rvs=Y.rvs)
                a = np.array(S1.A)
                b = np.array(S2.A)
                c = np.array(S3.A)
                d = np.array(S4.A)

            dottp = tp.linalg.multi_dot([a, b, c, d])
            dotnp = np.linalg.multi_dot([a, b, c, d])

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
            print(dotnp)
            print("numpy gets: ")
            print(dottp)
            errora = a
            errorb = b
            break

        if (dotnp == dottp).all():
            print(i, test_type + " correct")
            corr_num = corr_num + 1
        else:
            print("Calculation Error: " + test_type)
            print("Matrix a:\n ", a)
            print("Matrix b:\n ", b)
            print("transpy gets: ")
            print(dotnp)
            print("numpy gets: ")
            print(dottp)
            errora = a
            errorb = b
            break

    if test_num == corr_num:
        scoreSheet[test_type] = True
        print(test_num, "randomly-created cases tested. All Pass! Congrats!")

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
print("Testing multidot")
for i in scoreSheet.keys():
    if scoreSheet[i] == True:
        print(test_num, i, "cases tested. All Pass! Congrats!")
    else:
        print(i, "test failed. Pls Check!!!")
        print("Try:")
        print(errora)
        print(errorb)
