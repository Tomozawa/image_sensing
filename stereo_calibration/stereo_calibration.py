import xml.etree.ElementTree as ET
import cv2
import numpy as np
import re
import json
import datetime
from datetime import timezone, timedelta

class DescripterException(Exception):
    pass

class ImageProcessingException(Exception):
    pass

def calc_object_points(result, grid_len, row, col, zrot, xrot, yrot):
    chessboard_base_vectors = \
    np.array(
        [[np.cos(np.radians(zrot)), np.cos(np.radians(zrot + 90)), 0],
        [np.sin(np.radians(zrot)), np.sin(np.radians(zrot + 90)), 0],
        [0, 0, 1]]
    ) @ np.array(
        [[1, 0, 0],
        [0, np.cos(np.radians(xrot)), np.cos(np.radians(xrot + 90))],
        [0, np.sin(np.radians(xrot)), np.sin(np.radians(xrot + 90))]]
    ) @ np.array(
        [[np.sin(np.radians(yrot + 90)), 0, np.sin(np.radians(yrot))],
        [0, 1, 0],
        [np.cos(np.radians(yrot + 90)), 0, np.cos(np.radians(yrot))]]
    ) @ np.array(
        [[1, 0],
         [0, 1],
         [0, 0]]
    )

    object_points = []

    for i in range(0, row):
        for j in range(0, col):
            object_point = grid_len * chessboard_base_vectors @ np.array([[i], [j]])
            object_points.append(np.ravel(object_point.T).astype(np.float32))
    
    result['object_points'].append(np.array(object_points))

def find_image_points(result, img_file_name, row, col, is_left):
    src = cv2.imread(img_file_name)
    src_with_corners = np.copy(src)

    src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    if src.size == 0:
        raise ImageProcessingException(f'Failed to load {img_file_name}')
    print(f'{img_file_name} is loaded')
    
    ret, corners = cv2.findChessboardCorners(src, (col, row), None)

    if not ret:
        raise ImageProcessingException('failed to detection of control points')
    
    image_points = cv2.cornerSubPix(src, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

    cv2.drawChessboardCorners(src_with_corners, (col, row), image_points, True)
    cv2.imshow('calibration', src_with_corners)
    cv2.waitKey(0)

    print(f'succeed to detect control points: {len(image_points)}')

    image_points = [np.ravel(img_point) for img_point in image_points]
    image_points = np.array(image_points)
    result['left_image_points' if is_left else 'right_image_points'].append(image_points)

    return src.shape

def parse_img_tag(img_tag):
    limg_file_name = img_tag.attrib['lsrc']
    rimg_file_name = img_tag.attrib['rsrc']
        
    grid_len_str = img_tag.find('grid').attrib['len']
    match_list = re.findall('^([1-9][0-9]*)(m|cm|mm)?$', grid_len_str)
    if match_list == None or (len(match_list[0]) != 1 and len(match_list[0]) != 2):
        raise DescripterException(f'invalied len values: {match_list}')
    grid_len = int(match_list[0][0])
    if len(match_list[0]) == 2:
        grid_len *= 1000 if match_list[0][1] == 'm' else (10 if match_list[0][1] == 'cm' else 1)
    
    row = int(img_tag.find('grid/row').text)
    
    col = int(img_tag.find('grid/col').text)

    zrot = int(img_tag.find('posture/zrot').text)

    xrot = int(img_tag.find('posture/xrot').text)

    yrot = int(img_tag.find('posture/yrot').text)

    return (
        limg_file_name,
        rimg_file_name,
        grid_len,
        row,
        col,
        zrot,
        xrot,
        yrot
    )

def main():
###### calibration file diescriper format ######
# calibration file descripter is written in xml
# xml file must be names as "stereo_cal_description.xml"
# following tags are used to decribe calibration file
#
# <calibration type="stereo"> - the root tag
# <camera side="left or right"> - represents camera. It must contain <focal>
# <focal> - describe focal distance. The unit is mm. (decimal value is supported.)
# <sensor> - describe image sensor spec. It must contain <width> and <height>
# <width> - width of image sensor. The unit is mm. (decimal value is supported)
# <height> - height of image sensor. The unit is mm. (decimal value is suported)
# <img lsrc="path_to_img" rsrc="path_to_img"> - represents image file to be used for calibration. lsrc means the image taken by left camera, while rsrc does by right one. It must contains all the following tags
# <grid len="area length of grid"> - how many control points to find. m, cm, and mm are supportd as the unit of length(default unit is mm)
# <row> - num of vertical control points, included by <ctrlpts>
# <col> - num o horizontal control points, included by <ctrlpts>
# <posture> - posture of calibration object in euler expression(rotation order is z -> x -> y)
# <zrot> - centering on left-top control point, rotation in degree mesure around camera's optical axe, included by <posture>
# <xrot> - centering on left-top control point, rotation in degree mesure around right-heading axe, included by <posture>
# <yrot> - centering on left-top control point, rotation in degree mesure around bottom-heading axe(be careful with rotation direction), included by <posture>
#
###### example ######
# <?xml version="1.0" encoding="UTF-8" ?>
# <calibration type="stereo">
#   <camera side="left">
#       <focal>3</focal>
#       <sensor>
#           <width>3</width>
#           <height>3</height>
#       </sensor>
#   </camera>
#   <camera side="right">
#       <focal>3</focal>
#       <sensor>
#           <width>3</width>
#           <height>3</height>
#       </sensor>
#   </camera>
#   <img lsrc="./lcalibration.png" rsrc="./rcalibration.png">
#       <grid len="5cm">
#           <row>7</row>
#           <col>7</col>
#       </grid>
#       <posture>
#           <zrot>0</zrot>
#           <xrot>15</xrot>
#           <yrot>0</yrot>
#       </posture>
#   </img>
# </calibration>
    print('loading description file...')

    raw_string = ''

    with open('stereo_cal_description.xml', 'r', encoding='utf-8') as file:
        raw_string = file.read()
    
    root = ET.fromstring(raw_string)

    if root.tag != 'calibration' or root.attrib['type'] != 'stereo':
        raise DescripterException('invalied root tag name')
    
    corners = {
        'object_points': [],
        'left_image_points': [],
        'right_image_points': []
    }

    left_focal_length = float(root.find("camera[@side='left']/focal").text)

    right_focal_length = float(root.find("camera[@side='right']/focal").text)

    left_sensor_dimension = [
        float(root.find("camera[@side='left']/sensor/width").text),
        float(root.find("camera[@side='left']/sensor/height").text)
    ]

    right_sensor_dimension = [
        float(root.find("camera[@side='right']/sensor/width").text),
        float(root.find("camera[@side='right']/sensor/height").text)
    ]

    img_size = None

    for img in root.findall("img"):
        limg_file_name, rimg_file_name, grid_len, row, col, zrot, xrot, yrot = parse_img_tag(img)
        
        calc_object_points(corners, grid_len, row, col, zrot, xrot, yrot)

        try:
            limg_shape = find_image_points(corners, limg_file_name, row, col, True)
            rimg_shape = find_image_points(corners, rimg_file_name, row, col, False)
            if (not img_size) or (limg_shape[::-1] == img_size and rimg_shape[::-1] == img_size):
                img_size = limg_shape[::-1]
            else:
                raise ImageProcessingException('image size differs from each images of left camera')
        except ImageProcessingException as e:
            print(f'there was error in {limg_file_name} or {rimg_file_name}: {e}')
            exit(-1)
    
    left_initial_camera_matrix = np.array([
        [img_size[0] * left_focal_length / left_sensor_dimension[0], 0, img_size[0] / 2],
        [0, img_size[1] * left_focal_length / left_sensor_dimension[1], img_size[1] / 2],
        [0, 0, 1]
    ])

    right_initial_camera_matrix = np.array([
        [img_size[0] * right_focal_length / right_sensor_dimension[0], 0, img_size[0] / 2],
        [0, img_size[1] * right_focal_length / right_sensor_dimension[1], img_size[1] / 2],
        [0, 0, 1]
    ])

    ret, lmtx, ldist, _, _ = cv2.calibrateCamera(
        corners['object_points'],
        corners['left_image_points'],
        img_size,
        left_initial_camera_matrix,
        np.array([0, 0, 0, 0, 0]),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    print(f'lelf: {ret}')

    ret, rmtx, rdist, _, _ = cv2.calibrateCamera(
        corners['object_points'],
        corners['right_image_points'],
        img_size,
        right_initial_camera_matrix,
        np.array([0, 0, 0, 0, 0]),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )

    print(f'right: {ret}')

    ret, _, ldist, _, rdist, R, T, _, _ = cv2.stereoCalibrate(
        corners['object_points'],
        corners['left_image_points'],
        corners['right_image_points'],
        lmtx,
        ldist,
        rmtx,
        rdist,
        img_size,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f'relative position: {ret}')

    if not ret:
        raise ImageProcessingException('failed to calibration')
    
    timestamp = datetime.datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d%H%M%S')

    file_name = f'stereo_camera_calibration{timestamp}.json'

    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump({
            'left_matrix': lmtx.tolist(),
            'left_distortion': ldist.tolist(),
            'right_matrix': rmtx.tolist(),
            'right_distortion': rdist.tolist(),
            'R_matrix': R.tolist(),
            'T_vector': T.tolist(),
            'focal_distance': right_focal_length
        }, file)
    
    print(f'calibrationfile is saved to {file_name}')

if __name__ == '__main__':
    main()