'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2 as cv

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def original():
    print('loading images...')
    imgL = cv.pyrDown(cv.imread(cv.samples.findFile('left.jpg')))  # downscale images for faster processing
    imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR = cv.pyrDown(cv.imread(cv.samples.findFile('right.jpg')))
    imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1000,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')


def main():
    print('loading images...')
    imgL = cv.pyrDown(cv.pyrDown(cv.pyrDown(cv.imread(cv.samples.findFile('test_images/left3.jpg')))))  # downscale images for faster processing
    # imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR = cv.pyrDown(cv.pyrDown(cv.pyrDown(cv.imread(cv.samples.findFile('test_images/right3.jpg')))))
    # imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # disparity range is tuned for 'aloe' image pair
    # 2 * 3 * 2 * 1 * 3 * 1 * 3 * 2
    for min_disp in [26]:
        for num_disp in [16]:
            for blockSize in [6]:
                for disp12MaxDiff in [5]:
                    for uniquenessRatio in [5]:
                        for speckleWindowSize in [800]:
                            for speckleRange in [4]:
                                stereo = cv.StereoSGBM_create(
                                    minDisparity = min_disp,
                                    numDisparities = num_disp,
                                    blockSize = blockSize,
                                    P1 = 8*blockSize**2,
                                    P2 = 32*blockSize**2,
                                    disp12MaxDiff = disp12MaxDiff,
                                    uniquenessRatio = uniquenessRatio,
                                    speckleWindowSize = speckleWindowSize,
                                    speckleRange = speckleRange,
                                    mode = 3
                                )

                                print('computing disparity...')
                                disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

                                print('generating 3d point cloud...',)
                                h, w = imgL.shape[:2]
                                # f = focal_len_mm / sensor_width * im_width
                                f = 1.1*w
                                Q = np.float32([[1, 0, 0, -0.5*w],
                                                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                                                [0, 0, 0,     -f], # so that y-axis looks up
                                                [0, 0, 1,      0]])
                                points = cv.reprojectImageTo3D(disp, Q)
                                colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
                                mask = disp > disp.min()
                                out_points = points[mask]
                                out_colors = colors[mask]
                                out_fn = 'out_{}_{}_{}_{}_{}_{}_{}.ply'.format(min_disp, num_disp,
                                                                         blockSize, disp12MaxDiff,
                                                                         uniquenessRatio,
                                                                         speckleWindowSize,
                                                                         speckleRange)
                                write_ply(out_fn, out_points, out_colors)
                                cv.imwrite(out_fn.split(".")[0] + ".jpg", (255*((disp-min_disp)/num_disp)))
                                img = cv.imread(out_fn.split(".")[0] + ".jpg",0)
                                kernel = np.ones((5,5),np.float32)/25
                                dst = cv.filter2D(img,-1,kernel)

                                print('%s saved' % out_fn)

                                cv.imshow('left', imgL)
                                cv.imshow('disparity', (disp-min_disp)/num_disp)
                                cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
