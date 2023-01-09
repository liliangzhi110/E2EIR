import tensorflow as tf


def getPerspectiveTransformMatrix(input):

    batch=tf.shape(input)[0]

    point1 = tf.convert_to_tensor([[0, 0], [256, 0], [256, 256], [0, 256]],dtype=tf.float32)
    point1 = tf.reshape(point1, (1, 8))
    point1 = tf.tile(point1, [batch, 1])
    point2 = tf.subtract(point1,input)

    con = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)

    for i in range(0, batch):

        x1, x2, x3, x4 = point1[i:(i+1), 0][0], point1[i:(i+1), 2][0], point1[i:(i+1), 4][0], point1[i:(i+1), 6][0]
        y1, y2, y3, y4 = point1[i:(i+1), 1][0], point1[i:(i+1), 3][0], point1[i:(i+1), 5][0], point1[i:(i+1), 7][0]

        u1, u2, u3, u4 = point2[i:(i+1), 0][0], point2[i:(i+1), 2][0], point2[i:(i+1), 4][0], point2[i:(i+1), 6][0]
        v1, v2, v3, v4 = point2[i:(i+1), 1][0], point2[i:(i+1), 3][0], point2[i:(i+1), 5][0], point2[i:(i+1), 7][0]

        A = [[x1, y1, 1, 0, 0, 0, -u1 * x1, -u1 * y1, -u1],
            [0, 0, 0, x1, y1, 1, -v1 * x1, -v1 * y1, -v1],
            [x2, y2, 1, 0, 0, 0, -u2 * x2, -u2 * y2, -u2],
            [0, 0, 0, x2, y2, 1, -v2 * x2, -v2 * y2, -v2],
            [x3, y3, 1, 0, 0, 0, -u3 * x3, -u3 * y3, -u3],
            [0, 0, 0, x3, y3, 1, -v3 * x3, -v3 * y3, -v3],
            [x4, y4, 1, 0, 0, 0, -u4 * x4, -u4 * y4, -u4],
            [0, 0, 0, x4, y4, 1, -v4 * x4, -v4 * y4, -v4]]

        con.write(i,A)

    final_out = con.stack()
    U,S,Vh=tf.linalg.svd(final_out,full_matrices=True)
    Vh=tf.transpose(Vh)
    L = Vh[-1, :] / Vh[-1, -1]
    L=tf.transpose(L)

    return L





def Matrix(input):

    batch=tf.shape(input)[0]
    pointtemp1 = tf.convert_to_tensor([[0, 0], [500, 0], [500, 500], [0, 500]],dtype=tf.float32)
    pointtemp1 = tf.reshape(pointtemp1, (1, 8))
    pointtemp1 = tf.tile(pointtemp1, [batch, 1])
    pointtemp2 = tf.subtract(pointtemp1,input)

    point1=pointtemp2
    point2=pointtemp1

    x1, x2, x3, x4 = tf.reshape(point1[:, 0:1],(-1,1,1)), tf.reshape(point1[:, 2:3],(-1,1,1)), tf.reshape(point1[:, 4:5],(-1,1,1)), tf.reshape(point1[:, 6:7],(-1,1,1))
    y1, y2, y3, y4 = tf.reshape(point1[:, 1:2],(-1,1,1)), tf.reshape(point1[:, 3:4],(-1,1,1)), tf.reshape(point1[:, 5:6],(-1,1,1)), tf.reshape(point1[:, 7:8],(-1,1,1))

    u1, u2, u3, u4 = tf.reshape(point2[:, 0:1],(-1,1,1)), tf.reshape(point2[:, 2:3],(-1,1,1)), tf.reshape(point2[:, 4:5],(-1,1,1)), tf.reshape(point2[:, 6:7],(-1,1,1))
    v1, v2, v3, v4 = tf.reshape(point2[:, 1:2],(-1,1,1)), tf.reshape(point2[:, 3:4],(-1,1,1)), tf.reshape(point2[:, 5:6],(-1,1,1)), tf.reshape(point2[:, 7:8],(-1,1,1))

    #============================================第一点============================================
    oneline_x1pad0=tf.pad(x1,[[0,0],[0,7],[0,8]],constant_values=0)
    oneline_y1pad1 = tf.pad(y1, [[0, 0],[0,7], [1, 7]], constant_values=0)
    oneline_1pad2=tf.pad(tf.ones(shape=(batch,1,1),dtype=tf.float32),[[0, 0],[0,7], [2, 6]],constant_values=0)

    oneline_u1pad6=tf.pad(u1,[[0, 0],[0,7], [6, 2]],constant_values=0)
    oneline_x1pad6=tf.pad(x1,[[0, 0],[0,7], [6, 2]],constant_values=0)

    oneline_u1pad7=tf.pad(u1,[[0, 0],[0,7], [7, 1]],constant_values=0)
    oneline_y1pad7=tf.pad(y1,[[0, 0],[0,7], [7, 1]],constant_values=0)

    oneline_u1pad8=tf.pad(u1,[[0, 0],[0,7], [8, 0]],constant_values=0)



    twoline_x1pad3=tf.pad(x1,[[0,0],[1,6],[3,5]],constant_values=0)
    twoline_y1pad4=tf.pad(y1, [[0, 0],[1,6], [4, 4]], constant_values=0)
    twoline_1pad5 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0],[1,6], [5, 3]], constant_values=0)

    twoline_v1pad6=tf.pad(v1,[[0, 0],[1,6], [6, 2]],constant_values=0)
    twoline_x1pad6 = tf.pad(x1, [[0, 0], [1, 6], [6, 2]], constant_values=0)

    twoline_v1pad7 = tf.pad(v1, [[0, 0], [1, 6], [7, 1]], constant_values=0)
    twoline_y1pad7 = tf.pad(y1, [[0, 0], [1, 6], [7, 1]], constant_values=0)

    twoline_v1pad8 = tf.pad(v1, [[0, 0], [1, 6], [8, 0]], constant_values=0)

    con1=oneline_x1pad0+oneline_y1pad1+oneline_1pad2+(-oneline_u1pad6*oneline_x1pad6)+(-oneline_u1pad7*oneline_y1pad7)+(-oneline_u1pad8)
    con2=twoline_x1pad3+twoline_y1pad4+twoline_1pad5+(-twoline_v1pad6*twoline_x1pad6)+(-twoline_v1pad7*twoline_y1pad7)+(-twoline_v1pad8)



    # ============================================第2点============================================
    threeline_x2pad0 = tf.pad(x2, [[0, 0], [2, 5], [0, 8]], constant_values=0)
    threeline_y2pad1 = tf.pad(y2, [[0, 0], [2, 5], [1, 7]], constant_values=0)
    threeline_1pad2 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [2, 5], [2, 6]], constant_values=0)

    threeline_u2pad6 = tf.pad(u2, [[0, 0], [2, 5], [6, 2]], constant_values=0)
    threeline_x2pad6 = tf.pad(x2, [[0, 0], [2, 5], [6, 2]], constant_values=0)

    threeline_u2pad7 = tf.pad(u2, [[0, 0], [2, 5], [7, 1]], constant_values=0)
    threeline_y2pad7 = tf.pad(y2, [[0, 0], [2, 5], [7, 1]], constant_values=0)

    threeline_u2pad8 = tf.pad(u2, [[0, 0], [2, 5], [8, 0]], constant_values=0)



    fourline_x2pad3 = tf.pad(x2, [[0, 0], [3, 4], [3, 5]], constant_values=0)
    fourline_y2pad4 = tf.pad(y2, [[0, 0], [3, 4], [4, 4]], constant_values=0)
    fourline_1pad5 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [3, 4], [5, 3]], constant_values=0)

    fourline_v2pad6 = tf.pad(v2, [[0, 0], [3, 4], [6, 2]], constant_values=0)
    fourline_x2pad6 = tf.pad(x2, [[0, 0], [3, 4], [6, 2]], constant_values=0)

    fourline_v2pad7 = tf.pad(v2, [[0, 0], [3, 4], [7, 1]], constant_values=0)
    fourline_y2pad7 = tf.pad(y2, [[0, 0], [3, 4], [7, 1]], constant_values=0)

    fourline_v2pad8 = tf.pad(v2, [[0, 0], [3, 4], [8, 0]], constant_values=0)

    con3 = threeline_x2pad0+threeline_y2pad1+threeline_1pad2+(-threeline_u2pad6*threeline_x2pad6)+(-threeline_u2pad7*threeline_y2pad7)+(-threeline_u2pad8)
    con4=fourline_x2pad3+fourline_y2pad4+fourline_1pad5+(-fourline_v2pad6*fourline_x2pad6)+(-fourline_v2pad7*fourline_y2pad7)+(-fourline_v2pad8 )
    # ============================================第3点============================================
    fiveline_x3pad0 = tf.pad(x3, [[0, 0], [4, 3], [0, 8]], constant_values=0)
    fiveline_y3pad1 = tf.pad(y3, [[0, 0], [4, 3], [1, 7]], constant_values=0)
    fiveline_1pad2 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [4, 3], [2, 6]], constant_values=0)

    fiveline_u3pad6 = tf.pad(u3, [[0, 0], [4, 3], [6, 2]], constant_values=0)
    fiveline_x3pad6 = tf.pad(x3, [[0, 0], [4, 3], [6, 2]], constant_values=0)

    fiveline_u3pad7 = tf.pad(u3, [[0, 0], [4, 3], [7, 1]], constant_values=0)
    fiveline_y3pad7 = tf.pad(y3, [[0, 0], [4, 3], [7, 1]], constant_values=0)

    fiveline_u3pad8 = tf.pad(u3, [[0, 0], [4, 3], [8, 0]], constant_values=0)



    sixline_x3pad3 = tf.pad(x3, [[0, 0], [5, 2], [3, 5]], constant_values=0)
    sixline_y3pad4 = tf.pad(y3, [[0, 0], [5, 2], [4, 4]], constant_values=0)
    sixline_1pad5 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [5, 2], [5, 3]], constant_values=0)

    sixline_v3pad6 = tf.pad(v3, [[0, 0], [5, 2], [6, 2]], constant_values=0)
    sixline_x3pad6 = tf.pad(x3, [[0, 0], [5, 2], [6, 2]], constant_values=0)

    sixline_v3pad7 = tf.pad(v3, [[0, 0], [5, 2], [7, 1]], constant_values=0)
    sixline_y3pad7 = tf.pad(y3, [[0, 0], [5, 2], [7, 1]], constant_values=0)

    sixline_v3pad8 = tf.pad(v3, [[0, 0], [5, 2], [8, 0]], constant_values=0)

    con5 = fiveline_x3pad0 + fiveline_y3pad1 + fiveline_1pad2 + (-fiveline_u3pad6 * fiveline_x3pad6) + (
                -fiveline_u3pad7 * fiveline_y3pad7) + (-fiveline_u3pad8)
    con6= sixline_x3pad3 + sixline_y3pad4 + sixline_1pad5 + (-sixline_v3pad6 * sixline_x3pad6) + (
                -sixline_v3pad7 * sixline_y3pad7) + (-sixline_v3pad8)

    # ============================================第4点============================================
    sevenline_x4pad0 = tf.pad(x4, [[0, 0], [6, 1], [0, 8]], constant_values=0)
    sevenline_y4pad1 = tf.pad(y4, [[0, 0], [6, 1], [1, 7]], constant_values=0)
    sevenline_1pad2 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [6, 1], [2, 6]], constant_values=0)

    sevenline_u4pad6 = tf.pad(u4, [[0, 0], [6, 1], [6, 2]], constant_values=0)
    sevenline_x4pad6 = tf.pad(x4, [[0, 0], [6, 1], [6, 2]], constant_values=0)

    sevenline_u4pad7 = tf.pad(u4, [[0, 0], [6, 1], [7, 1]], constant_values=0)
    sevenline_y4pad7 = tf.pad(y4, [[0, 0], [6, 1], [7, 1]], constant_values=0)

    sevenline_u4pad8 = tf.pad(u4, [[0, 0], [6, 1], [8, 0]], constant_values=0)



    eightline_x4pad3 = tf.pad(x4, [[0, 0], [7, 0], [3, 5]], constant_values=0)
    eightline_y4pad4 = tf.pad(y4, [[0, 0], [7, 0], [4, 4]], constant_values=0)
    eightline_1pad5 = tf.pad(tf.ones(shape=(batch, 1,1), dtype=tf.float32), [[0, 0], [7, 0], [5, 3]], constant_values=0)

    eightline_v4pad6 = tf.pad(v4, [[0, 0], [7, 0], [6, 2]], constant_values=0)
    eightline_x4pad6 = tf.pad(x4, [[0, 0], [7, 0], [6, 2]], constant_values=0)

    eightline_v4pad7 = tf.pad(v4, [[0, 0], [7, 0], [7, 1]], constant_values=0)
    eightline_y4pad7 = tf.pad(y4, [[0, 0], [7, 0], [7, 1]], constant_values=0)

    eightline_v4pad8 = tf.pad(v4, [[0, 0], [7, 0], [8, 0]], constant_values=0)

    con7 = sevenline_x4pad0 + sevenline_y4pad1 + sevenline_1pad2 + (-sevenline_u4pad6 * sevenline_x4pad6) + (
            -sevenline_u4pad7 * sevenline_y4pad7) + (-sevenline_u4pad8)
    con8 = eightline_x4pad3 + eightline_y4pad4 + eightline_1pad5 + (-eightline_v4pad6 * eightline_x4pad6) + (
            -eightline_v4pad7 * eightline_y4pad7) + (-eightline_v4pad8)

    #=============================================计算=============================================
    con_con=con1+con2+con3+con4+con5+con6+con7+con8

    U,S,Vh=tf.linalg.svd(con_con,full_matrices=True)
    Vh=tf.transpose(Vh)
    L = Vh[-1, :] / Vh[-1, -1]
    L=tf.transpose(L)


    return L

















































# # @tf.function
# # def test(x):
# #     tensor_list = []
# #     for i in tf.range(x):
# #         tensor_list.append(tf.ones(4)*tf.cast(i, tf.float32))
# #     return  tf.stack(tensor_list)
# #
# # result = test(5)
# # print(result)
#
#
# input_ta = tf.TensorArray(size=0, dtype=tf.int32, dynamic_size=True)
#
#
# @tf.function
# def test(x):
#
#     tensor_list = tf.map_fn(lambda inp: tf.ones(4)*tf.cast(inp, tf.float32), x, dtype=tf.dtypes.float32)
#
#     return  tf.stack(tensor_list)
#
# result = test(np.arange(5))
# print(result)
#
#
#
#
#
#
# tensor_list = tf.map_fn(lambda inp: tf.ones(4)*tf.cast(inp, tf.float32), np, dtype=tf.dtypes.float32)
#
#
#
#
#
#
# @tf.function
# def getPerspectiveTransformMatrix(input):
#
#     batch=tf.shape(input)[0]
#
#     point1 = tf.convert_to_tensor([[0, 0], [256, 0], [256, 256], [0, 256]],dtype=tf.float32)
#     point1 = tf.reshape(point1, (1, 8))
#     point1 = tf.tile(point1, [batch, 1])
#     point2 = tf.subtract(point1,input)
#
#
#     zero=tf.zeros(shape=(1,))[0]
#     one=tf.zeros(shape=(1,))[0]+1
#
#
#     batch_A=[]
#     for i in range(0, batch):
#
#         x1, x2, x3, x4 = point1[i:(i+1), 0][0], point1[i:(i+1), 2][0], point1[i:(i+1), 4][0], point1[i:(i+1), 6][0]
#         y1, y2, y3, y4 = point1[i:(i+1), 1][0], point1[i:(i+1), 3][0], point1[i:(i+1), 5][0], point1[i:(i+1), 7][0]
#
#         u1, u2, u3, u4 = point2[i:(i+1), 0][0], point2[i:(i+1), 2][0], point2[i:(i+1), 4][0], point2[i:(i+1), 6][0]
#         v1, v2, v3, v4 = point2[i:(i+1), 1][0], point2[i:(i+1), 3][0], point2[i:(i+1), 5][0], point2[i:(i+1), 7][0]
#
#         # A = [[x1, y1, 1, 0, 0, 0, -u1 * x1, -u1 * y1, -u1],
#         #      [0, 0, 0, x1, y1, 1, -v1 * x1, -v1 * y1, -v1],
#         #      [x2, y2, 1, 0, 0, 0, -u2 * x2, -u2 * y2, -u2],
#         #      [0, 0, 0, x2, y2, 1, -v2 * x2, -v2 * y2, -v2],
#         #      [x3, y3, 1, 0, 0, 0, -u3 * x3, -u3 * y3, -u3],
#         #      [0, 0, 0, x3, y3, 1, -v3 * x3, -v3 * y3, -v3],
#         #      [x4, y4, 1, 0, 0, 0, -u4 * x4, -u4 * y4, -u4],
#         #      [0, 0, 0, x4, y4, 1, -v4 * x4, -v4 * y4, -v4]]
#
#         A = [[x1, y1, one, zero, zero, zero, -u1 * x1, -u1 * y1, -u1],
#              [zero, zero, zero, x1, y1, one, -v1 * x1, -v1 * y1, -v1],
#              [x2, y2, one, zero, zero, zero, -u2 * x2, -u2 * y2, -u2],
#              [zero, zero, zero, x2, y2, one, -v2 * x2, -v2 * y2, -v2],
#              [x3, y3, one, zero, zero, zero, -u3 * x3, -u3 * y3, -u3],
#              [zero, zero, zero, x3, y3, one, -v3 * x3, -v3 * y3, -v3],
#              [x4, y4, one, zero, zero, zero, -u4 * x4, -u4 * y4, -u4],
#              [zero, zero, zero, x4, y4, one, -v4 * x4, -v4 * y4, -v4]]
#
#         batch_A.append(A)
#
#
#
#     print('00000000000000000000000000')
#     batch_A=tf.stack(batch_A)
#
#     arrayA=tf.reshape(batch_A,(batch,8,9))
#     # print(batch_A, '[[[[[[[[[[[[[[[[[')
#     # print('test')
#
#     U,S,Vh=tf.linalg.svd(arrayA,full_matrices=True)
#     Vh=tf.transpose(Vh)
#     L = Vh[-1, :] / Vh[-1, -1]
#     L=tf.transpose(L)
#
#     return L









