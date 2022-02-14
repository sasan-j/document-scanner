import random
import numpy as np
import cv2
import json
import pathlib


def setup_logger(path):
    import logging

    logger = logging.getLogger("zebel-scanner")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(path / "log.txt")
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class CustomEncoder(json.JSONEncoder):
    def default(self, x):
        if isinstance(x, pathlib.Path):
            return str(x)
        else:
            return super().default(x)


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


def __get_cords(cord, min_start, max_end, size=299, buf=5, random_scale=True):
    # size = max(abs(cord-min_start), abs(cord-max_end))
    iter = 0
    if random_scale:
        size /= random.randint(1, 4)
    while (max_end - min_start) < size:
        size = size * 0.9
    temp = -1
    while temp < 1:
        temp = random.normalvariate(size / 2, size / 6)
    x_start = max(cord - temp, min_start)
    x_start = int(x_start)
    if x_start >= cord:
        print("XSTART AND CORD", x_start, cord)
    assert x_start < cord
    while (
        (x_start < min_start) or (x_start + size > max_end) or (x_start + size <= cord)
    ):
        # x_start = random.randint(int(min(max(min_start, int(cord - size + buf)), cord - buf - 1)), cord - buf)
        temp = -1
        while temp < 1:
            temp = random.normalvariate(size / 2, size / 6)
        temp = max(temp, 1)
        x_start = max(cord - temp, min_start)
        x_start = int(x_start)
        size = size * 0.995
        iter += 1
        if iter == 1000:
            x_start = int(cord - (size / 2))
            print("Gets here")
            break
    assert x_start >= 0
    if x_start >= cord:
        print("XSTART AND CORD", x_start, cord)
    assert x_start < cord
    assert x_start + size <= max_end
    assert x_start + size > cord
    return (x_start, int(x_start + size))


def get_corners(img, gt):
    gt = gt.astype(int)
    list_of_points = {}
    myGt = gt

    myGtTemp = myGt * myGt
    sum_array = myGtTemp.sum(axis=1)

    tl_index = np.argmin(sum_array)
    tl = myGt[tl_index]
    tr = myGt[(tl_index + 1) % 4]
    br = myGt[(tl_index + 2) % 4]
    bl = myGt[(tl_index + 3) % 4]

    list_of_points["tr"] = tr
    list_of_points["tl"] = tl
    list_of_points["br"] = br
    list_of_points["bl"] = bl
    gt_list = []
    images_list = []
    for k, v in list_of_points.items():

        if k == "tl":
            cords_x = __get_cords(
                v[0],
                0,
                list_of_points["tr"][0],
                buf=10,
                size=abs(list_of_points["tr"][0] - v[0]),
            )
            cords_y = __get_cords(
                v[1],
                0,
                list_of_points["bl"][1],
                buf=10,
                size=abs(list_of_points["bl"][1] - v[1]),
            )
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0] : cords_y[1], cords_x[0] : cords_x[1]]

        if k == "tr":
            cords_x = __get_cords(
                v[0],
                list_of_points["tl"][0],
                img.shape[1],
                buf=10,
                size=abs(list_of_points["tl"][0] - v[0]),
            )
            cords_y = __get_cords(
                v[1],
                0,
                list_of_points["br"][1],
                buf=10,
                size=abs(list_of_points["br"][1] - v[1]),
            )
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0] : cords_y[1], cords_x[0] : cords_x[1]]

        if k == "bl":
            cords_x = __get_cords(
                v[0],
                0,
                list_of_points["br"][0],
                buf=10,
                size=abs(list_of_points["br"][0] - v[0]),
            )
            cords_y = __get_cords(
                v[1],
                list_of_points["tl"][1],
                img.shape[0],
                buf=10,
                size=abs(list_of_points["tl"][1] - v[1]),
            )
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0] : cords_y[1], cords_x[0] : cords_x[1]]

        if k == "br":
            cords_x = __get_cords(
                v[0],
                list_of_points["bl"][0],
                img.shape[1],
                buf=10,
                size=abs(list_of_points["bl"][0] - v[0]),
            )
            cords_y = __get_cords(
                v[1],
                list_of_points["tr"][1],
                img.shape[0],
                buf=10,
                size=abs(list_of_points["tr"][1] - v[1]),
            )
            # print cords_y, cords_x
            gt = (v[0] - cords_x[0], v[1] - cords_y[0])

            cut_image = img[cords_y[0] : cords_y[1], cords_x[0] : cords_x[1]]

        # cv2.circle(cut_image, gt, 2, (255, 0, 0), 6)
        mah_size = cut_image.shape
        cut_image = cv2.resize(cut_image, (300, 300))
        a = int(gt[0] * 300 / mah_size[1])
        b = int(gt[1] * 300 / mah_size[0])
        images_list.append(cut_image)
        gt_list.append((a, b))
    return images_list, gt_list
