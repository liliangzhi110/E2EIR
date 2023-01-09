import numpy as np
import cv2
def numpy_conv(inputs,filter,padding="VALID"):
    H, W = inputs.shape
    filter_size = filter.shape[0]
    # default np.floor
    filter_center = int(filter_size / 2.0)
    filter_center_ceil = int(np.ceil(filter_size / 2.0))

    # SAME模式，输入和输出大小一致，所以要在外面填充0
    if padding == "SAME":
        padding_inputs = np.zeros([H + filter_center_ceil, W + filter_center_ceil], np.float32)
        padding_inputs[filter_center:-filter_center, filter_center:-filter_center] = inputs
        inputs = padding_inputs
    #这里先定义一个和输入一样的大空间，但是周围一圈后面会截掉
    result = np.zeros((inputs.shape))
    #更新下新输入,SAME模式下，会改变HW
    H, W = inputs.shape
    #print("new size",H,W)
    #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
    for r in range(filter_center,H -filter_center):
        for c in range(filter_center,W -filter_center ):
            #获取卷积核大小的输入区域
            cur_input = inputs[r -filter_center :r +filter_center_ceil,
                        c - filter_center:c + filter_center_ceil]
            #和核进行乘法计算
            cur_output = cur_input * filter
            #再把所有值求和
            conv_sum = np.sum(cur_output)
            #当前点输出值
            result[r, c] = conv_sum
    # 外面一圈都是0，裁减掉
    final_result = result[filter_center:result.shape[0] - filter_center,
                   filter_center:result.shape[1] -filter_center]
    return final_result



image_cv2_band7=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band1.tif",cv2.IMREAD_UNCHANGED)
image_cv2_band1=cv2.imread("D:\\Second_paper_VAE_CNN\\xian_image\\xi_an_sub10000band3.tif",cv2.IMREAD_UNCHANGED)


image_cv2_band7=image_cv2_band7[0:100,0:100]
image_cv2_band1=image_cv2_band1[0:100,0:100]

filt=np.array([
    [0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,1,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0]
])


im7=numpy_conv(image_cv2_band7,filt)
im1=numpy_conv(image_cv2_band1,filt)



print('直接减法',np.sum(image_cv2_band7-image_cv2_band1))
print('相似性计算',np.sum(im7-im1))

































