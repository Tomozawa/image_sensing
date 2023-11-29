import xml.etree.ElementTree as ET
import cv2
import re
import numpy as np
import json
from datetime import timezone, timedelta
import datetime

class DescripterException(Exception):
    pass

class ImageProcessingExcepton(Exception):
    pass

hsv_param = {
    "hmin": 0,
    "hmax": 0,
    "smin": 0,
    "smax": 0,
    "vmin": 0,
    "vmax": 0
}

img = None
bin_img = None

def hmin_callback(val):
    hsv_param['hmin'] = val
    execute_calc()

def hmax_callback(val):
    hsv_param['hmax'] = val
    execute_calc()

def smin_callback(val):
    hsv_param['smin'] = val
    execute_calc()

def smax_callback(val):
    hsv_param['smax'] = val
    execute_calc()

def vmin_callback(val):
    hsv_param['vmin'] = val
    execute_calc()

def vmax_callback(val):
    hsv_param['vmax'] = val
    execute_calc()

def initialize_param_img():
    global img_tag, bin_img
    hsv_param['hmin'] = 0
    hsv_param['hmax'] = 0
    hsv_param['smin'] = 0
    hsv_param['smax'] = 0
    hsv_param['vmin'] = 0
    hsv_param['vmax'] = 0
    img_tag = None
    bin_img = None

def execute_calc():
    global bin_img
    copied_img = np.copy(img)
    copied_img = cv2.GaussianBlur(copied_img, (11, 11), 8.5)
    h = copied_img[..., 0]
    s = copied_img[..., 1]
    v = copied_img[..., 2]

    h = cv2.inRange(
        h,
        min(hsv_param['hmin'], hsv_param['hmax']),
        max(hsv_param['hmin'], hsv_param['hmax'])
    )
    s = cv2.inRange(
        s,
        hsv_param['smin'],
        hsv_param['smax']
    )
    v = cv2.inRange(
        v,
        hsv_param['vmin'],
        hsv_param['vmax']
    )

    if hsv_param['hmin'] > hsv_param['hmax']:
        h = cv2.bitwise_not(h)
    
    hsv_filtered = cv2.bitwise_and(s, v)
    hsv_filtered = cv2.bitwise_and(hsv_filtered, h)

    opening_closing_times = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (-1, -1))
    hsv_filtered = cv2.erode(
        hsv_filtered,
        kernel,
        iterations=opening_closing_times
    )
    hsv_filtered = cv2.dilate(
        hsv_filtered,
        kernel,
        iterations=opening_closing_times
    )
    hsv_filtered = cv2.dilate(
        hsv_filtered,
        kernel,
        iterations=opening_closing_times
    )
    hsv_filtered = cv2.erode(
        hsv_filtered,
        kernel,
        iterations=opening_closing_times
    )

    bin_img = hsv_filtered

def main():
###### scaling file descripter format ######
# scaling file descripter is written in xml
# sml file mus be named as "sca_description.xml"
# following tags are used to describe scaling file
#
# <scaling> - the root tag
# <img src="path_to_img"> - represents imege file to be used for scaling. It must contains all the following tags
# <distance> - distance from optical origin to object. m,cm, and mm are supported as the unit of length(default unit is mm)
#
###### examle ######
# <?xml version="1.0" encoding="UTF-8" ?>
# <scaling>
#   <img src="scaling1.jpg">
#       <distance>100mm<distance>
#   </img>
#   <img src="scaling2.jpg">
#       <distance>10cm</distance>
#   </img>
# </scaling>
    global img
    print('loading description file...')
    raw_string = ''

    with open('sca_description.xml', 'r', encoding='utf-8') as file:
        raw_string = file.read()
    
    root = ET.fromstring(raw_string)

    if root.tag != 'scaling':
        raise DescripterException('invalied root tag name')
    
    distances = []
    areas = []

    for img_tag in root.findall('img'):
        initialize_param_img()

        img_file_name = img_tag.attrib['src']
        
        img = cv2.imread(img_file_name)

        distance_str = img_tag.find('distance').text
        match_list = re.findall('^([1-9][0-9]*)(m|cm|mm)?$', distance_str)
        if match_list == None or (len(match_list[0]) != 1 and len(match_list[0]) != 2):
            raise DescripterException(f'invalied distance values: {match_list}')
        distance = int(match_list[0][0])
        print(distance)
        if len(match_list[0]) == 2:
            distance *= 1000 if match_list[0][1] == 'm' else (10 if match_list[0][1] == 'cm' else 1)
            print(1000 if match_list[0][1] == 'm' else (10 if match_list[0][1] == 'cm' else 1))

        window_name = f'scaling: {img_file_name}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            'Hue min',
            window_name,
            0,
            255,
            hmin_callback
        )
        cv2.createTrackbar(
            'Hue max',
            window_name,
            0,
            255,
            hmax_callback
        )
        cv2.createTrackbar(
            'Saturation min',
            window_name,
            0,
            255,
            smin_callback
        )
        cv2.createTrackbar(
            'Saturation max',
            window_name,
            0,
            255,
            smax_callback
        )
        cv2.createTrackbar(
            'Value min',
            window_name,
            0,
            255,
            vmin_callback
        )
        cv2.createTrackbar(
            'Value max',
            window_name,
            0,
            255,
            vmax_callback
        )

        cv2.imshow(img_file_name, img)

        execute_calc()

        while(cv2.waitKey(1) == -1):
            cv2.imshow(window_name, bin_img)
        
        cv2.destroyAllWindows()
        
        print('parameter is saved')

        canny = cv2.Canny(bin_img, 25, 75)

        contours, hierarchys = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = map(lambda val: cv2.convexHull(cv2.approxPolyDP(val, 0.005 * cv2.arcLength(val, True), True)), contours)

        contours = [contour for (contour, hierarchy) in zip(contours, hierarchys[0]) if cv2.contourArea(contour) >= 400 and hierarchy[3] < 0]

        if len(contours) != 1:
            print('multiple contours are detected. tell me which contours you want to use?')
            select_contour_image = np.copy(img)
            colorlist = [
                ((0, 255, 0), 'green'),
                ((0, 0, 255), 'red'),
                ((255, 0, 0), 'blue'),
                ((0, 255, 255), 'yellow'),
                ((255, 0, 255), 'magenta'),
                ((255, 255, 0), 'cyan')
            ]
            if len(contours) > len(colorlist):
                print(f'available contrours number is up to {len(colorlist)}, while I detected {len(contours)}. some contours are ignored.')
            
            for (index, (_, color)) in enumerate(zip(contours, colorlist)):
                cv2.drawContours(select_contour_image, contours, index, color[0])
                print(f'contour{index + 1}: {color[1]}')
            
            cv2.imshow(window_name, select_contour_image)

            print('continue to press any key...')

            cv2.waitKey(0)

            cv2.destroyAllWindows()

            selected_index = 0
            while(not 1 <= selected_index <= len(list(zip(contours, colorlist)))):
                selected_index_str = input('tell me the index of contour>>')
                if not selected_index_str.isdigit():
                    continue
                selected_index = int(selected_index_str)
            
            contour = contours[selected_index - 1]
        
        print(cv2.contourArea(contour))

        areas.append(cv2.contourArea(contour))
        distances.append(distance)

    print(distances)

    a = sum([float(distance) ** -4 for distance in distances])
    b = sum([area / (distance ** 2) for (area, distance) in zip(areas, distances)])

    timestamp = datetime.datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d%H%M%S')

    with open(f'camera_scaling_{timestamp}.json', 'w', encoding='utf-8') as file:
        json.dump(
            {
                "area": b / a,
                "distance": 1
            },
            file
        )
    
    print(f'scaling file is saved as camera_scaling_{timestamp}.json')

if __name__ == '__main__':
    main()