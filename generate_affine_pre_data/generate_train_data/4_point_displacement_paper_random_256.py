import cv2
import numpy as np

one=np.load("D:\\Second_paper_VAE_CNN\\xi_an_landsat8\\one\\landsat1_1.npz")['image'][0]
second=np.load("D:\\Second_paper_VAE_CNN\\xi_an_landsat8\\second\\landsat2_2.npz")['image'][0]

one=(one-np.min(one))/(np.max(one)-np.min(one))
second=(second-np.min(second))/(np.max(second)-np.min(second))
one=one.astype('float32')
second=second.astype('float32')

fixed0=[]
moving0=[]
displacement_4_point0=[]


for x in range(256,one.shape[0]-256,32):
    for y in range(256,one.shape[1]-256,64):
        print(x,y)

        sub_fixed_512=one[x-256:x+256,y-256:y+256]
        sub_moved_512 = second[x - 256:x + 256, y - 256:y + 256]

        sub_fixed = sub_fixed_512[128:384, 128:384]

        fixed0.append(sub_fixed.reshape((256, 256, 1)))

        points1 = np.float32([[128, 128], [384, 128], [384, 384], [128, 384]])
        points2 = np.float32([[np.random.randint(118,138,size=(1,))[0],np.random.randint(118,138,size=(1,))[0]],
                              [np.random.randint(374,394,size=(1,))[0], np.random.randint(118,138,size=(1,))[0]],
                              [np.random.randint(374,394,size=(1,))[0], np.random.randint(374,394,size=(1,))[0]],
                              [np.random.randint(118,138,size=(1,))[0], np.random.randint(374,394,size=(1,))[0]]])

        wrap_moving_matrix=cv2.getPerspectiveTransform(points2,points1)

        moving_image=cv2.warpPerspective(sub_moved_512,wrap_moving_matrix,(512,512))

        sub_moving_image=moving_image[128:384,128:384]

        moving0.append(sub_moving_image.reshape((256, 256, 1)))

        temp=points1-points2

        displacement_4_point0.append(temp.reshape(1,8))


fixed0=np.array(fixed0)
moving0=np.array(moving0)
displacement_4_point0=np.array(displacement_4_point0)


fixed1=[]
moving1=[]
displacement_4_point1=[]


for x in range(256,one.shape[0]-256,64):
    for y in range(256,one.shape[1]-256,64):
        print(x,y)

        sub_fixed_512=one[x-256:x+256,y-256:y+256]
        sub_moved_512 = second[x - 256:x + 256, y - 256:y + 256]

        sub_fixed = sub_fixed_512[128:384, 128:384]

        fixed1.append(sub_fixed.reshape((256, 256, 1)))

        points1 = np.float32([[128, 128], [384, 128], [384, 384], [128, 384]])
        points2 = np.float32([[np.random.randint(118,138,size=(1,))[0],np.random.randint(118,138,size=(1,))[0]],
                              [np.random.randint(374,394,size=(1,))[0], np.random.randint(118,138,size=(1,))[0]],
                              [np.random.randint(374,394,size=(1,))[0], np.random.randint(374,394,size=(1,))[0]],
                              [np.random.randint(118,138,size=(1,))[0], np.random.randint(374,394,size=(1,))[0]]])

        wrap_moving_matrix=cv2.getPerspectiveTransform(points2,points1)

        moving_image=cv2.warpPerspective(sub_moved_512,wrap_moving_matrix,(512,512))

        sub_moving_image=moving_image[128:384,128:384]

        moving1.append(sub_moving_image.reshape((256, 256, 1)))

        temp=points1-points2

        displacement_4_point1.append(temp.reshape(1,8))


fixed1=np.array(fixed1)
moving1=np.array(moving1)
displacement_4_point1=np.array(displacement_4_point1)



fixed2=[]
moving2=[]
displacement_4_point2=[]


for x in range(256,one.shape[0]-256,64):
    for y in range(256,one.shape[1]-256,64):
        print(x,y)

        sub_fixed_512=one[x-256:x+256,y-256:y+256]
        sub_moved_512 = second[x - 256:x + 256, y - 256:y + 256]

        sub_fixed = sub_fixed_512[128:384, 128:384]

        fixed2.append(sub_fixed.reshape((256, 256, 1)))

        points1 = np.float32([[128, 128], [384, 128], [384, 384], [128, 384]])
        points2 = np.float32([[np.random.randint(118,138,size=(1,))[0],np.random.randint(118,138,size=(1,))[0]],
                              [np.random.randint(374,394,size=(1,))[0], np.random.randint(118,138,size=(1,))[0]],
                              [np.random.randint(374,394,size=(1,))[0], np.random.randint(374,394,size=(1,))[0]],
                              [np.random.randint(118,138,size=(1,))[0], np.random.randint(374,394,size=(1,))[0]]])

        wrap_moving_matrix=cv2.getPerspectiveTransform(points2,points1)

        moving_image=cv2.warpPerspective(sub_moved_512,wrap_moving_matrix,(512,512))

        sub_moving_image=moving_image[128:384,128:384]

        moving2.append(sub_moving_image.reshape((256, 256, 1)))

        temp=points1-points2

        displacement_4_point2.append(temp.reshape(1,8))


fixed2=np.array(fixed2)
moving2=np.array(moving2)
displacement_4_point2=np.array(displacement_4_point2)



fixed=np.concatenate((fixed0,fixed1,fixed2),axis=0)
moving=np.concatenate((moving0,moving1,moving2),axis=0)
displacement_4_point=np.concatenate((displacement_4_point0,displacement_4_point1,displacement_4_point2),axis=0)





np.savez("D:\\ProgramData_second\\generate_affine_pre_data\\data\\generated_npz_image\\displacement_4_point_paper_random_landsat_256.npz",
         fixed=fixed,
         moving=moving,
         displacement_4_point=displacement_4_point)