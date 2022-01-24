import numpy as np
import cv2


def sort_gt(gt):
    """
    Sort the ground truth labels so that TL corresponds to the label with smallest distance from O
    :param gt:
    :return: sorted gt
    """
    myGtTemp = gt * gt
    sum_array = myGtTemp.sum(axis=1)
    tl_index = np.argmin(sum_array)
    tl = gt[tl_index]
    tr = gt[(tl_index + 1) % 4]
    br = gt[(tl_index + 2) % 4]
    bl = gt[(tl_index + 3) % 4]

    return np.asarray((tl, tr, br, bl))


def __rotateImage(image, angle):
    rot_mat = cv2.getRotationMatrix2D(
        (image.shape[1] / 2, image.shape[0] / 2), angle, 1
    )
    result = cv2.warpAffine(
        image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR
    )
    return result, rot_mat


def rotate(img, gt, angle):
    img, mat = __rotateImage(img, angle)
    gt = gt.astype(np.float64)
    for a in range(0, 4):
        gt[a] = np.dot(mat[..., 0:2], gt[a]) + mat[..., 2]
    return img, gt


def random_crop(img, gt):
    ptr1 = (
        min(gt[0][0], gt[1][0], gt[2][0], gt[3][0]),
        min(gt[0][1], gt[1][1], gt[2][1], gt[3][1]),
    )

    ptr2 = (
        max(gt[0][0], gt[1][0], gt[2][0], gt[3][0]),
        max(gt[0][1], gt[1][1], gt[2][1], gt[3][1]),
    )

    start_x = np.random.randint(0, int(max(ptr1[0] - 1, 1)))
    start_y = np.random.randint(0, int(max(ptr1[1] - 1, 1)))

    end_x = np.random.randint(int(min(ptr2[0] + 1, img.shape[1] - 1)), img.shape[1])
    end_y = np.random.randint(int(min(ptr2[1] + 1, img.shape[0] - 1)), img.shape[0])

    img = img[start_y:end_y, start_x:end_x]

    myGt = gt - (start_x, start_y)
    myGt = myGt * (1.0 / img.shape[1], 1.0 / img.shape[0])

    myGtTemp = myGt * myGt
    sum_array = myGtTemp.sum(axis=1)
    tl_index = np.argmin(sum_array)
    tl = myGt[tl_index]
    tr = myGt[(tl_index + 1) % 4]
    br = myGt[(tl_index + 2) % 4]
    bl = myGt[(tl_index + 3) % 4]

    return img, (tl, tr, br, bl)
