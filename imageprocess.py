import numpy as np
import cv2 as cv
import glob
import math #在坐标转换中使用
from collections import Counter #用来对块进行颜色分类
from math import sqrt#算平方根
from math import radians#角度变弧度
import abb
from pyquaternion import Quaternion #用来做四元数转换

def calibrate_getmtx(folderpath):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    wnum=11
    hnum=8
    objp = np.zeros((wnum*hnum,3), np.float32)
    objp[:,:2] = np.mgrid[0:wnum,0:hnum].T.reshape(-1,2)
    # arrays to store object points and image points from all the images.
    objpoints=[] # 3d points in the real world
    imgpoints=[] # 2d points in image plane
    images=glob.glob("{}/*.png".format(folderpath))
    for fname in images:
        img=cv.imread(fname)
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (wnum,hnum),None,cv.CALIB_CB_ADAPTIVE_THRESH)

        if ret==True:
            objpoints.append(objp)
            corners2=cv.cornerSubPix(gray,corners,(8,8),(-1,-1),criteria)
            imgpoints.append(corners)
            print("find in {}".format(fname))
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx,dist,rvecs # return the intrinsic camera matrixs

def ouput_undistort_image_path(image_path,mtx, dist):# refine matrixs and undistort the image
    image=cv.imread("{}".format(image_path))
    h, w =image.shape[:2]# what this shape exactly means?
    newcameramtx, roi =cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst=cv.undistort(image, mtx, dist, None, newcameramtx)
    x,y,w,h=roi
    dst=dst[y:y+h,x:x+w]
    cv.imwrite('/home/yang/projects/project_from_labor/CSU-robot-arm/calibresult.png',dst)
    print("the image has been output to the path:",'/home/yang/projects/project_from_labor/CSU-robot-arm/calibresult.png')
    path='/home/yang/projects/project_from_labor/CSU-robot-arm/calibresult.png'
    return path,newcameramtx

def compute_angle(vec1,vec2):
    unit_vector_1 = vec1/ np. linalg. norm(vec1)
    unit_vector_2 = vec2/np.linalg.norm(vec2)
    dot_prod=np. dot(unit_vector_1, unit_vector_2)
    angle=np. arccos(dot_product)
    return angle
    
def get_contours_and_center(img_path,w=1,isgray=False,isauto=True):# input img,return a list containing all the block information needed for stacking and its width
    image=cv.imread(img_path)

    imgray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)

    trans_image=hsv[:,:,2]
    if isgray:
        trans_image=imgray
    if isauto:
        thre1=cv.adaptiveThreshold(trans_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,30)
    else:
        ret1,thre1=cv.threshold(imgray,0,255,cv.THRESH_BINARY)
    mask=thre1

    black=cv.bitwise_and(hsv,hsv,mask=mask)
    black_gray=cv.cvtColor(black,cv.COLOR_HSV2BGR)
    black_gray=cv.cvtColor(black_gray,cv.COLOR_BGR2GRAY)

    _,thre1=cv.threshold(black_gray,10,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    img_morph=cv.morphologyEx(thre1,cv.MORPH_OPEN,kernel=(3,3))
    img_morph=cv.erode(img_morph,(3,3),img_morph,iterations=2)
    morph=cv.dilate(img_morph,(3,3),img_morph,iterations=2)
    cv.imwrite("/home/yang/projects/project_from_labor/CSU-robot-arm/morph.jpg",morph)
    contours, hierachy =cv.findContours(morph, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    array1=contours
    
    # select one contour that meet the standard of rectangle
    flag3=0
    for i in array1:
        if flag3==0:
            leftmost = tuple(i[i[:,:,0].argmin()][0])
            rightmost = tuple(i[i[:,:,0].argmax()][0])
            topmost = tuple(i[i[:,:,1].argmin()][0])
            bottommost = tuple(i[i[:,:,1].argmax()][0])

            main_cross=(leftmost[0]-rightmost[0])**2+(leftmost[1]-rightmost[1])**2
            vice_cross=(topmost[0]-bottommost[0])**2+(topmost[1]-bottommost[1])**2

            if abs(main_cross-vice_cross)<100 and cv.contourArea(i)<4500 and cv.contourArea(i)>200:
                desired_contour=i
                flag3=1
                print("obtained a rectangle")
                pre_select=cv.drawContours(image,i, -1, (0, 255, 0), 2)
                cv.imshow("desired contour",pre_select)
                cv.waitKey(0)
                cv.destroyAllWindows()
                
        else:
            break
    
    list3=[]
    j=0 #calculator of the number of iteration
    for a in array1:
        area=cv.contourArea(a)# find the area

        ret=cv.matchShapes(desired_contour,a,1,0.0)

        if area<4500 and area>400 and ret < 0.15:
            M=cv.moments(a)

            #get the mean color:mask and use mean function
            mask1=np.zeros(black_gray.shape,dtype='uint8' )
            cv.drawContours(mask1,a,-1,255,-1)
            mean=cv.mean(black_gray,mask=mask1)

            #detect the rotation of the rect
            rotated_rect=cv.minAreaRect(a)#It returns a Box2D structure which contains following detals - ( top-left corner(x,y), (width, height), angle of rotation )

            if rotated_rect[1][0]>rotated_rect[1][1]:
                angle=rotated_rect[2]
            
            else:
                angle=rotated_rect[2]+90                        #加90度保证一致性 （？）逻辑有问题？
            
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # print("points are {},{}".format(cX,cY)) 测试cx，cy是像素坐标还是图像坐标
                cv.circle(image, (cX,cY), 7, (0, 0, 0), -1)
                img_withcontour=cv.drawContours(image,a, -1,(0,255,0),3)
                img_withcontour=cv.drawContours(image,a, -1, (0, 255, 0), 2)

                #通过输出角度判断
                text1="{}".format(int(mean[0]))

                cv.putText(image,text1,(cX,cY),cv.FONT_HERSHEY_COMPLEX,1.0, (100, 200, 200), 3)
            else:
                print("m00 is zero")

            #用来计算宽度的部分，日后可以独立出去作为一个单独的函数
            width_esti=[]
            line1=(leftmost[0]-topmost[0])**2+(leftmost[1]-topmost[1])**2
            line2=(leftmost[0]-bottommost[0])**2+(leftmost[1]-bottommost[1])**2
            if line1<line2:
                width_esti=[leftmost,topmost]
            else:
                width_esti=[leftmost,bottommost]
            
            j=j+1
            props=[mean,cX,cY,angle]
            print(mean)
            list3.append(props)
   

    cv.imshow("img",img_withcontour)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return list3, width_esti
# folderpath="/home/yang/projects/fname1

def colour_categorization_and_sorting(list4):
    #对积木按照颜色进行分类并且按照数量从多到少进行排列
    '''
    colornum=0
    # 为所有赋值
    for i in range(len(list4)):
        if len(i)==4:
            colornum=colornum+1
            list4(i).append(colornum)
            for j in range(i,len(list4)):
                if abs(i[0]-j[0])<1:
                    list[j].append(colornum)
    # 根据颜色数字分类
    '''
    # 整理
    colorlist=[]
    for i in range(len(list4)):
        colorlist.append(int(list4[i][0][0]))

    dict1=Counter(colorlist)

    sorted_list=[]
    for items in dict1.items():
        color=items[0]
        for j in range(len(list4)):
            int_color=int(list4[j][0][0])
            if int_color==color:
                sorted_list.append(list4[j])

    #规划堆叠方式
    stack_info=[]
    if len(list4)==7:
        stack_info=[3,2,2]
    
    return sorted_list,stack_info

def width_estimation(var1,revc,mtx):#利用像素世界中矩形短边的两点算出在世界坐标中矩形的大小
    x1,y1=trans_to_world(var1[0],revc,mtx)
    x2,y2=trans_to_world(var1[1],revc,mtx)
    width=sqrt((x1-x2)**2+(y1-y2)**2)
    return width


def trans_to_world(center,rvecs,mtx,worldx0=659,worldy0=527,Zc=1420):
    obj_x,obj_y=center[0],center[1]    
    fx,fy=mtx[0][0],mtx[1][1]
    dxc=Zc/fx*abs(obj_x-worldx0)
    dyc=Zc/fy*abs(obj_y-worldy0)
    numrvecs=len(rvecs)
    transx,transy,transz=0,0,0

    for i in range(0,numrvecs-1):
        transx=transx+rvecs[i][0]
        transy=transy+rvecs[i][1]
        transz=transz+rvecs[i][2]

    transx=math.radians(transx/numrvecs)
    transy=math.radians(transy/numrvecs)
    transz=math.radians(transz/numrvecs)

    Rx=np.array([[1,0,0],[0,math.cos(transx),math.sin(transx)],[0,-math.sin(transx),math.cos(transx)]])
    Ry=np.array([[math.cos(transy),0,math.sin(transy)],[0,1,0],[-math.sin(transy),0,math.cos(transy)]])
    Rz=np.array([[math.cos(transz),math.sin(transz),0],[-math.sin(transz),math.cos(transz),0],[0,0,1]])

    trans=np.dot(np.dot(Rx,Ry),Rz)
    dpoint=np.array([dxc,dyc,1])
    dpoint=dpoint.T
    dw=np.dot(trans,dpoint)
    dxw,dyw=dw[0],dw[1]

    if obj_x-worldx0<=0:
        dxw=dxw
    else:
        dxw=-dxw
    if obj_y-worldy0<=0:
        dyw=dyw
    else:
        dyw=-dyw
    return dxw,dyw

# def stack_planning(blocknum,blockcolor):
def robot_move(thing,st_list,width,obj_point,worldx0=400,worldy0=400,ip1='192.168.125.1'):
    R=abb.Robot(ip=ip1)
    R.set_joints([0,0,0,0,0,0])
    #设置tool down
    q=Quaternion([0,0,1,0])
    j1=0
    sum_j=0#用来表示现在是第几个
    height=15#代表积木高度
    for j in st_list:
        j1=j1+1#代表层数
        k=0#k代表本层第几
        for i in range(j):
            index=sum_j+i
            # 将角度换为弧度
            angle1=radians(thing[index][3])
            # 换算四元数
            my_quaternion = Quaternion(axis=[0,0,1], angle=angle1)
            quat=(q*my_quaternion).elements
            #手臂到达其上空
            R.set_cartesian([[thing[index][1]+worldx0,thing[index][2]+worldy0,200],quat])
            #抓取
            R.set_cartesian([[thing[index][1]+worldx0,thing[index][2]+worldy0,height],quat])
            #抬起
            R.set_cartesian([[thing[index][1]+worldx0,thing[index][2]+worldy0,200],[quat[0],quat[1],quat[2],quat[3]]])

            quat=Quaternion(axis=[0,0,1],angle=-angle1)
            quat=(q*quat).elements
            #平移
            R.set_cartesian([[obj_point[0]+width*k,obj_point[1],200],[quat[0],quat[1],quat[2],quat[3]]])
            #放置
            R.set_cartesian([[obj_point[0]+width*k,obj_point[1],(j1)*height],[quat[0],quat[1],quat[2],quat[3]]])
            print(R.get_cartesian())
            #提起来
            R.set_cartesian([[obj_point[0]+width*k,obj_point[1],200],[quat[0],quat[1],quat[2],quat[3]]])
            k=k+1
        sum_j=sum_j+j




def main():
    folderpath="/home/yang/projects/camera1"
    imagepath="/home/yang/projects/camera1/c40fef4a612c3c62d9bb78c030a33be.jpg"
    #main
    mtx,dist,rvec=calibrate_getmtx(folderpath)
    newimagepath,mtx1=ouput_undistort_image_path(imagepath,mtx,dist)
    obj_list,width_esti=get_contours_and_center(newimagepath)
    so_list,st_list=colour_categorization_and_sorting(obj_list)
    width=width_estimation(width_esti,rvec,mtx1)
    print('the width is {}'.format(width))

    # 从sortedlist中找到每个cxcy，进行坐标转换，放回list中
    j=0#j是用来计算他是图像中第几个图片的
    for i in so_list:
        cX,cY,angle=i[1],i[2],i[3]
        centre=[cX,cY]
        wX,wY=trans_to_world(centre,rvec,mtx1)
        i[1],i[2]=wX,wY
    robot_move(so_list,st_list,width,obj_point=[400,-400])

    
    



main()

