import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LightSource
import math 
import scipy 
import cp_hw2
import cp_hw6
import skimage

from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray
import os
import glob 

def readimage(path):
    return cv2.imread(path).astype(np.float32) / 255.0


def showimage(image, cm='gray'):
    plt.imshow(image, cmap=cm)
    plt.show()

def writeimage(path, image):
    skimage.io.imsave(path, image)

def readfrogs(gray = True):
    Istack = list()
    # N = 166
    N = 221
    for i in range(N):
        file = str(i + 1)
        while len(file) < 6:
            file = '0' + file 
        I = readimage('../data/guoba/' + file + '.jpg')
        if gray == True: 
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        else:
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        Istack.append(I)
    return Istack



def plotline(l, r):
    a = l[0]
    b = l[1]
    c = l[2]
    points = list()
    for x in range(r[2], r[3]):
        y = (a * x + c) / (-b)
        if y >= r[0] and y < r[1]:
            points.append(np.array([x,y]))
    
    if len(points) < 20:
        points = list()
        for y in range(r[0], r[1]):
            x = (b * y + c) / (-a)
            if x >= r[2] and x < r[3]:
                points.append(np.array([x,y]))

    x = np.array(points)[:,0]
    y = np.array(points)[:,1]
    plt.plot(x, y)

def get_twopoints(l, r):
    a = l[0]
    b = l[1]
    c = l[2]
    points = list()
    for x in range(r[2], r[3]):
        y = (a * x + c) / (-b)
        if y >= r[0] and y < r[1]:
            points.append(np.array([x,y]))
    
    if len(points) < 20:
        points = list()
        for y in range(r[0], r[1]):
            x = (b * y + c) / (-a)
            if x >= r[2] and x < r[3]:
                points.append(np.array([x,y]))
    
    pa = points[0]
    pb = points[len(points) - 1]
    return np.array([0.1 * pa + 0.9 * pb, 0.9 * pa + 0.1 * pb])

# frog
# v_reg = [0,300,250,780]
# h_reg = [655,765,200,820]

# cabi
# v_reg = [180,285,522, 1300]
# h_reg = [784,955,365,1300]

# guoba
v_reg = [125,250,553,1455]
h_reg = [900,1000,464,1400]

def process():
    Istack = readfrogs()
    dim = Istack[0].shape

    # showimage(Istack[0])
    # print(dim)
    dstack = np.dstack(Istack)
    Imin = np.min(dstack, axis=2)
    Imax = np.max(dstack, axis=2)
    Ishadow = (Imin + Imax)/2.0
    

    def compute_shadow_time():
        t_sha = np.zeros(dim)
        mx_sha = np.zeros(dim)
        last_Idiff = np.zeros(dim)
        eps = 1e-6
        thres = 0.01
        for t in range(len(Istack)):
            I = Istack[t]
            Idiff = I - Ishadow
            d = Idiff - last_Idiff
            cross = (t_sha <= eps) & (Idiff > 0) & (last_Idiff < 0)
            t_sha += cross * t
            last_Idiff = Idiff
        return t_sha

    t_sha = compute_shadow_time()
    # showimage(t_sha, 'jet')
    plt.imsave('guoba_shadow_time.jpg', t_sha, cmap = 'jet')
    np.save('shadow_time.npy', t_sha)



    def compute_shadow_edges(r):
        cnt = 0
        edges = dict()
        for t in range(20,170):
            I = Istack[t]
            Idiff = I - Ishadow
            split_points = list()
            for row in range(r[0], r[1]):
                has = False 
                n_col = 0
                for col in range(r[2], r[3]):
                    u = Idiff[row][col-1]
                    v = Idiff[row][col]
                    if u < 0 and v > 0:
                        has = True 
                        n_col = col
                        # print(v - u)
                if has == True:
                    split_points.append([n_col, row, 1])

            # if(len(split_points) != 0):
            #     print(t, len(split_points))
            #     plt.imshow(I, cmap='gray')
            #     x = np.array(split_points)[:,0]
            #     y = np.array(split_points)[:,1]
            #     plt.plot(x, y)
            #     plt.show()
            
            if len(split_points) > 10:
                A = np.array(split_points)
                u,s,vh = np.linalg.svd(A, full_matrices=False)
                x = -vh[-1]
                res = A @ x
                edges[t] = (x[0],x[1],x[2])
            cnt+=1
        return edges

    vedges = compute_shadow_edges(v_reg)
    hedges = compute_shadow_edges(h_reg)
    vp = list()
    hp = list()
    for t in vedges:
        if t in hedges:
            v = vedges[t]
            h = hedges[t]
            vp.append(np.array([t,v[0],v[1],v[2]]))
            hp.append(np.array([t,h[0],h[1],h[2]]))
    
    numedges = len(vp)
    np.save('vp.npy', np.array(vp))
    np.save('hp.npy', np.array(hp))
    # print('numedges : ', numedges)
    for i in range(numedges):
        vedge = vp[i]
        hedge = hp[i]
        
        t = int(vedge[0])
        print(t, vedge, hedge)
        I = Istack[t]
        # if t % 10 == 0:
        #     print(t)
        #     plt.imshow(I, cmap = 'gray')
        #     plotline((vedge[1],vedge[2],vedge[3]), v_reg)
        #     plotline((hedge[1],hedge[2],hedge[3]), h_reg)
        #     plt.show()
    print(len(vedges))
    print(len(hedges))

    #showimage(t_sha, 'jet')

def camera_to_plane(R,T,camera):
    return R.transpose() @ (camera - T)

def plane_to_camera(R,T,plane):
    return (np.linalg.inv(R.transpose()) @ plane) + T


def intersect(r, plane):
    p1 = plane[0]
    n = plane[1]

    # np.dot(n, x) + d = 0
    d = -np.dot(n, p1)
    p = np.array([0,0,0])
    t = -(np.dot(n, p) + d) / np.dot(n, r)
    return p + t * r



def calib():
    #Input data locations
    baseDir = '../data' #data directory
    objName = 'cabi' #object name (should correspond to a dir in data)
    seqName = 'v1' #sequence name (subdirectory of object)
    calName = 'calib_final' #calibration sourse (also a dir in data)
    image_ext = 'jpg' #file extension for images
    useLowRes = False #enable lowres for debugging

    #Extrinsic calibration parameters
    dW1 = (8, 8) #window size for finding checkerboard corners
    checkerboard = (6, 8) #number of internal corners on checkerboard

    #Intrinsic calibration parameters
    dX = 558.8 #calibration plane length in x direction
    dY = 303.2125 #calibration plane length in y direction
    dW2 = (8, 8) #window size finding ground plane corners

    if useLowRes:
        calName += '-lr'
        seqName += '-lr'

    #Part 1: Intrinsic Calibration

    # images = glob.glob(os.path.join(baseDir, calName, "*"+image_ext))
    # mtx, dist = computeIntrinsic(images, checkerboard, dW1)
    # #write out intrinsic calibration parameters
    # np.savez(os.path.join(baseDir, calName, "intrinsic_calib.npz"), mtx=mtx, dist=dist)

    # #Part 2: Extrinsic Calibration
    # #load intrinsic parameters
    with np.load(os.path.join(baseDir, calName, "intrinsic_calib.npz")) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
    
    # print(mtx, dist)


    # #obtain extrinsic calibration from reference plane

    # firstFrame = os.path.join(baseDir, objName, seqName, '000001.jpg')
    # print("Perform horizontal extrinsic calibration")
    # tvec_h, rmat_h = computeExtrinsic(firstFrame, mtx, dist, dX, dY)

    # print("Perform vertical extrinsic calibration")
    # tvec_v, rmat_v = computeExtrinsic(firstFrame, mtx, dist, dX, dY)

    # ext_out = {"tvec_h":tvec_h, "rmat_h":rmat_h, "tvec_v":tvec_v, "rmat_v":rmat_v}
    # np.savez(os.path.join(baseDir, objName, seqName, "extrinsic_calib.npz"), **ext_out) 
    with np.load(os.path.join(baseDir, objName, seqName, "extrinsic_calib.npz")) as X:
        tvec_h, rmat_h = X['tvec_h'], X['rmat_h']
        tvec_v, rmat_v = X['tvec_v'], X['rmat_v']
        print(tvec_h, rmat_h)
        print(tvec_v, rmat_v)

    Istack = readfrogs()
    vp = np.load('vp.npy')
    hp = np.load('hp.npy')


    def get_vertical_P_new(p):
        r = p.transpose()
        p0 = np.array([[0],[0],[0]])
        p1 = np.array([[0],[0],[1]])
        new_p0 = plane_to_camera(rmat_v, tvec_v, p0)
        new_p1 = plane_to_camera(rmat_v, tvec_v, p1)

        n = (new_p1 - new_p0).flatten() 
        n = n/np.linalg.norm(n)
        res = intersect(r.flatten(), (new_p0.flatten(),n.flatten()))
        return res

    def get_horizontal_P_new(p):
        r = p.transpose()
        p0 = np.array([[0],[0],[0]])
        p1 = np.array([[0],[0],[1]])
        new_p0 = plane_to_camera(rmat_h, tvec_h, p0)
        new_p1 = plane_to_camera(rmat_h, tvec_h, p1)
        n = (new_p1 - new_p0).flatten() 
        n = n/np.linalg.norm(n)
        res = intersect(r.flatten(), (new_p0.flatten(),n.flatten()))
        return res

    t_list = list()
    p1_list = list()
    points_list = list()
    n_list = list()

    for i in range(len(vp)):
        l,r = vp[i],hp[i]
        t = l[0]
        # print(t)
        
        
        px = get_twopoints((l[1],l[2],l[3]), v_reg)
        py = get_twopoints((r[1],r[2],r[3]), h_reg)

        # if int(t) % 5 == 0:
        #     plt.imshow(Istack[int(t)], cmap = 'gray')
        #     plt.plot(np.array([px[0][0],px[1][0]]),np.array([px[0][1],px[1][1]]))
        #     plt.plot(np.array([py[0][0],py[1][0]]),np.array([py[0][1],py[1][1]]))
        #     plt.show()

        # print(px.shape)

    	# in camera coordinates
        rx = pixel2ray(px, mtx, dist)
        ry = pixel2ray(py, mtx, dist)

        r1 = rx[0]
        r2 = rx[1]
        r3 = ry[0]
        r4 = ry[1]

        p1 = get_vertical_P_new(r1).flatten()
        p2 = get_vertical_P_new(r2).flatten()
        p3 = get_horizontal_P_new(r3).flatten()
        p4 = get_horizontal_P_new(r4).flatten()

        n = np.cross(p2 - p1, p4 - p3) 

        if(np.linalg.norm(n) == 0):
            continue
        # normalize
        n = n/ np.linalg.norm(n)

        t_list.append(t)
        p1_list.append(p1)
        n_list.append(n)
        points_list.append(np.array([p1,p2,p3,p4]))


        # if int(t) % 5 == 0:
        #     print(t)
        #     po = np.array([np.array([0,0,0]),p1,p2,p3,p4])
        #     ax = plt.axes(projection='3d')
        #     ax.scatter3D(po[:,0], po[:,1], po[:,2], c=po[:,2]);
        #     plt.show()

        # print(p1,p2,p3,p4)

        # print(rmat_h.shape, tvec_h.shape, r1.shape)
        # o = camera_to_plane(rmat_v, tvec_v, np.array([[0],[0],[0]]))
        # v1 = camera_to_plane(rmat_v, tvec_v, r1.transpose())
        # v2 = camera_to_plane(rmat_v, tvec_v, r2.transpose())

        # print(o)
        # print(v1)


        # h1 = camera_to_plane(rmat_h, tvec_h, r3.transpose())
        # h2 = camera_to_plane(rmat_h, tvec_h, r4.transpose())
        # print(v1)

        # print("rx : ", rx)
        # print("ry : ", ry)
        # plt.plot(p[:,0], p[:,1])
        # #print(p[0],p[1])

        # # plotline((l[1],l[2],l[3]), v_reg)
        # # plotline((r[1],r[2],r[3]), h_reg)
        # plt.show()
        
    # print(vp)

    ext_out = {"frame_t": np.array(t_list), "four points" : np.array(points_list)}
    np.savez("reconstruct_points.npz", **ext_out)

    ext_out = {"frame_t": np.array(t_list), "p1":np.array(p1_list), "n":np.array(n_list)}
    np.savez("shadow_plane.npz", **ext_out)

def reconstruct():
    # frog
    # roi = [[317,652],[289,808]]
    # cabi
    # roi = [[346,788],[780,1190]]
    # guoba
    roi = [[267,900],[615,1401]]

    def sub(I, reg):
        return I[reg[0][0]:reg[0][1],reg[1][0]:reg[1][1]]

    st = np.load("shadow_time.npy").astype(np.dtype('int32'))
    # plt.imshow(st, cmap = 'jet')
    plt.imshow(sub(st, roi), cmap = 'jet')
    plt.show()

    Istack = readfrogs()
    dstack = np.dstack(Istack)

    Imin = np.min(dstack, axis = 2)
    Imax = np.max(dstack, axis = 2)
    Isub = Imax - Imin

    Icolorstack = np.array(readfrogs(False))


    R = np.mean(Icolorstack[:,:,:,0], axis = 0)
    G = np.mean(Icolorstack[:,:,:,1], axis = 0)
    B = np.mean(Icolorstack[:,:,:,2], axis = 0)
    Inoshadow = np.dstack([R,G,B])
    showimage(sub(Inoshadow, roi))

    # plt.imshow(Inoshadow)
    # plt.show()

    # print(np.min(sub(Isub, roi)))
    thres = 0.05


    with np.load("intrinsic_calib.npz") as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
    with np.load("shadow_plane.npz") as X:
        frame_set,p1_set,n_set = X['frame_t'],X['p1'],X['n']

    s_dict = dict()
    for i in range(len(frame_set)):
        s_dict[int(frame_set[i])] = (p1_set[i], n_set[i])



    
    plist = list()
    flist = list()

    for y in range(roi[0][0], roi[0][1]):
        for x in range(roi[1][0], roi[1][1]):
            if Isub[y][x] > thres:
                r = pixel2ray(np.array([[float(x),float(y)]]), mtx, dist)
                t = int(st[y][x])
                
                if t in s_dict:
                    plane = s_dict[t]
                    p = intersect(r[0][0], plane)
                    if(np.max(np.abs(p)) >= 2500):
                        continue
                    plist.append(p)
                    # print(p)
                    flist.append(Inoshadow[y][x])
    plist = np.array(plist)
    flist = np.array(flist)
    percent = len(plist)/((roi[1][1] - roi[1][0]) * (roi[0][1] - roi[0][0]))
    print("percentage : {:f}%".format(percent * 100.0))

    np.save("plist.npy", plist)
    np.save("flist.npy", flist)



def showscatter(plist, flist):
    o = np.array([0.0,0.0,0.0])
    d = 0.0 + len(plist)
    for p in plist:
        o += p / d
    dlist = list()
    for p in plist:
        n = p - o
        dlist.append(np.linalg.norm(n))
    dlist = np.array(dlist)
    new_plist = list()
    new_flist = list()
    u = np.mean(dlist)
    t = np.std(dlist)
    for i in range(len(plist)):
        d = dlist[i]
        if np.abs(d - u) <= 1.5 * t:
            new_plist.append(plist[i])
            new_flist.append(flist[i])
    plist = np.array(new_plist)
    flist = np.array(new_flist)



    plist = plist[::1]
    flist = flist[::1]

    # print(plist.shape)
    # print(np.min(plist), np.max(plist))
            
    ax = plt.axes(projection='3d')
    ax.scatter3D(plist[:,0], plist[:,1], plist[:,2], c=flist);
    ax.view_init(83, 90)
    plt.show()


def main():
    # part 1.1
    
    # process()
    
    # part 1.2
    # calib()

    # part 1.3
    reconstruct()

    plist = np.load('plist.npy')
    flist = np.load('flist.npy')
    showscatter(plist, flist)
main()