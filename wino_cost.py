import tensorflow as tf
import numpy as np
import wincnn
from pprint import pprint
from sympy import Rational
from collections import Counter

INPUT = np.random.rand(1, 227, 227, 256)
WEIGHT = np.random.rand(3, 3, 256, 256)

M = 4
R = 3
W = M + R - 1
interpolations = (0, 1, -1, 2, -2, Rational(1, 2), -Rational(1, 2),
                  Rational(1, 4), -Rational(1, 4))

def conv1d_tf(inputs_data, weight_data):
    '''conv1d by tensorflow'''
    inputs = tf.placeholder(tf.float32, shape=(1, 256, 256))
    weight = tf.placeholder(tf.float32, shape=(3, 256, 256))
    output = tf.nn.conv1d(inputs, weight, 1, 'VALID', data_format='NCHW')

    sess = tf.Session()
    result_tf = sess.run(output, feed_dict={inputs: inputs_data,
                                            weight: weight_data})
    return result_tf

def findCostByRow(M):
    mat = np.abs(np.array(M).astype(np.float64))
    mult_n = 0
    add_n = np.count_nonzero(mat) - mat.shape[0]
    
    for vec in mat:
        counter = Counter()
        for x in vec:
            counter[x] += 1
        for x, _ in counter.items():
            if x == 0.0:
                continue
            else:
                mult_n = mult_n if x == 1.0 else mult_n + 1
    return mult_n, add_n

def costWinograd1D():
    cost = {}
    for itp in range(4, np.size(interpolations) + 1):
        for m in range(2, itp - 1):
            r = itp - m
            l = interpolations[:m+r-2]
            # print(m, r, l)
            AT, G, BT, _ = wincnn.cookToomFilter(l, m, r)
            cost[("%d %d" % (m, r))] = [findCostByRow(AT), 
                                        findCostByRow(G), 
                                        findCostByRow(BT),
                                        (m + r -1, m + r - 1),
                                        (m * r, m * r)]
    pprint(cost)

def costWinograd2D():
    cost = {}
    for itp in range(4, np.size(interpolations) + 1):
        for m in range(2, itp - 1):
            r = itp - m
            w = m + r - 1
            l = interpolations[:m+r-2]
            AT, G, BT, _ = wincnn.cookToomFilter(l, m, r)
            cost[("%d %d" % (m, r))] = [np.multiply(findCostByRow(AT), w + m), 
                                        np.multiply(findCostByRow(G), w + r), 
                                        np.multiply(findCostByRow(BT), 2 * w),
                                        (w**2, w**2),
                                        (r*r*m*m , r*r*m*m)]
    pprint(cost)

costWinograd2D()

#result = arithComplxWino1D(INPUT, WEIGHT, AT, G, BT, 'SAME')
# print(result)
# wincnn.showCookToomFilter((0, 1, -1, Rational(3, 1), -Rational(3, 1)), M, R)

# RESULT_TF = conv1d_tf(INPUT, WEIGHT)

# def conv1d_wg(inputs_data, weight_data, AT, G, BT):
#     ''' conv1d by winograd
#         Y = AT x [(G x g) dot (BT x d)]
#     '''
#     trans_width = M + R - 1
#     batch = inputs_data.shape[0]
#     cin = inputs_data.shape[1]
#     width = inputs_data.shape[2]
#     cout = weight_data.shape[2]

#     print(trans_width, batch, channels, width)
#     temp = []
    
#     for n in range(0, batch):
#         for p in range(0, width - trans_width, M):
#             for c2 in range(cout):
#                 for c1 in range(0, cin):
#                     trans_input = inputs_data[n][cin][p : p + trans_width]
#                     filter = weight[0][cin]
#                     trans_weight = weight[]
#                     temp[c].append(trans_input)

#     return temp


# def arithComplxWino1D(data, weight, AT, G, BT, padding):
#     '''
#         input   : input feature map in NHWC order
#         weight  : weight/filter in H W Cin Cout order
#         AT      : inverse transformation matrix (transposed)
#         G       : filter transformation matrix
#         BT      : input transformation matrix
#         padding : 'SAME' or 'VALID' as defined by Tensorflow   

#         stride  : default to 1, cannot change

#         multN   : total number of multiplications
#         addN    : total number of additions
#         ref.    : http://arxiv.org/abs/1701.03534
#     '''
    
#     # Input  : n x cin x h x w
#     # Filter : cout x cin x r x s
#     # Output : n x cout x h x w
#     n = data.shape[0]
#     h = data.shape[1]
#     w = data.shape[2]
    
#     r = weight.shape[0]
#     s = weight.shape[1]
#     cout = weight.shape[2]
#     cin = weight.shape[3]

#     p = h if padding == 'SAME' else h - r + 1
#     q = w if padding == 'SAME' else w - s + 1

    
#     # Input 1D vector size
#     w_vec = BT.shape[1]
#     # Output 1D vector size
#     q_vec = w_vec - r + 1
    
#     # Data transform BT x d, per input vector
#     multN_dt, addN_dt = findCostByRow(BT)
#     # Transforming weights, assuming g is 1D
#     multN_ft, addN_ft = findCostByRow(G)
#     # Inverse transform AT x (BT x d) * (G x g)
#     multN_it, addN_it = findCostByRow(AT)
#     # Effective dot product
#     multN_dp, addN_dp = w_vec, w_vec

#     print(multN_dt, multN_ft, multN_it, multN_dp, s * q_vec)
#     print(addN_dt, addN_ft, addN_it, addN_dp, s * q_vec)
#     # Total transforms needed to get one output vector

#     multN = [cin * s * multN_dp,
#              cin * s * multN_ft, 
#              cin * s * multN_dt, 
#              multN_it]
#     addN = [cin * s * addN_dp,
#             cin * s * addN_ft, 
#             cin * s * addN_dt, 
#             addN_it]
    
#     # Total number of output vectors

#     outVecN = p * np.ceil(q / q_vec) * cout * n
    
#     multN = np.multiply(multN, outVecN)
#     addN = np.multiply(addN, outVecN)
    
#     origin = p * q * s * r * cin * cout
    
#     return multN, addN, origin