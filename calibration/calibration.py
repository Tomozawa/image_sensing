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

def find_corners(result, img_file_name, grid_len, row, col, zrot, xrot, yrot):
    src = cv2.imread(img_file_name, cv2.IMREAD_GRAYSCALE)
    if src.size == 0:
        raise ImageProcessingException(f'Failed to load {img_file_name}')
    print(f'{img_file_name} is loaded')

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

    print('detecing controllpoints...')
    
    ret, corners = cv2.findChessboardCorners(src, (row, col), None)

    if not ret:
        raise ImageProcessingException('failed to detection of control points')
    
    image_points = cv2.cornerSubPix(src, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

    print(f'succeed to detect control points: {len(image_points)}')

    image_points = [np.ravel(img_point) for img_point in image_points]
    image_points = np.array(image_points)
    result['image_points'].append(image_points)

    result['object_points'].append(np.array(object_points))

    return src.shape

def main():
###### calibration file descripter format ######
# calibration file descripter is written in xml
# xml file must be named as "cal_description.xml"
# following tags are used to describe cakibration file
#
# <calibration> - the root tag
# <img src="path_to_img"> - represents image file to be used for calibration. It must contains all the following tags
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
# <calibration>
#   <img src="./calibration.png">
#       <grid len="5cm">
#           <row>7</row>
#           <col>7</col>
#       </grid>
#       <posture>
#           <zrot>0</zrot>
#           <xrot>15</zrot>
#           <yrot>0</yrot>
#       </posture>
#   </img>
# </calibration>
    print('loading descripiton file...')

    row_string = ''

    with open('cal_description.xml', 'r', encoding='utf-8') as file:
        row_string = file.read()
    
    root = ET.fromstring(row_string)

    if root.tag != 'calibration':
        raise DescripterException('invalied root tag name')
    
    corners = {
        'object_points': [],
        'image_points': []
    }

    img_size = None

    for img in root:
        img_file_name = img.attrib['src']
        
        grid_len_str = img.find('grid').attrib['len']
        match_list = re.findall('^([1-9][0-9]*)(m|cm|mm)?$', grid_len_str)
        if match_list == None or (len(match_list[0]) != 1 and len(match_list[0]) != 2):
            raise DescripterException(f'invalied len values: ${match_list}')
        grid_len = int(match_list[0][0])
        if len(match_list[0]) == 2:
            grid_len *= (not match_list[0][1] == 'm') if ((not match_list[0][1] == 'cm') if 1 else 10) else 1000
        
        row = int(img.find('grid/row').text)
        
        col = int(img.find('grid/col').text)

        zrot = int(img.find('posture/zrot').text)

        xrot = int(img.find('posture/xrot').text)

        yrot = int(img.find('posture/yrot').text)

        try:
            img_shape = find_corners(corners, img_file_name, grid_len, row, col, zrot, xrot, yrot)
            if (not img_size) or img_shape[::-1] == img_size:
                img_size = img_shape[::-1]
            else:
                raise ImageProcessingException('image size differs from each images')
        except ImageProcessingException as e:
            print(f'there was error in {img_file_name}: {e}')

    for image_points_in_view in corners['image_points']:
        if len(image_points_in_view) < 15:
            raise ImageProcessingException(f'there are too few control points: {len(corners["image_points"])}')

    ret, mtx, dist, _, _ = cv2.calibrateCamera(corners['object_points'], corners['image_points'], img_size, None, None)

    if not ret:
        raise ImageProcessingException('failed to calibration')
    
    timestamp = datetime.datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d%H%M%S')

    file_name = f'camera_calibration{timestamp}.json'
    
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump({
            'matrix': mtx.tolist(),
            'distortion': dist.tolist()
        }, file)

    print(f'caribration file is saved to {file_name}')

if __name__ == '__main__':
    main()