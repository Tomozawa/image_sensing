import json
from math import pi
import datetime
from datetime import timedelta, timezone

def main():
    calibration = None
    with open('camera_calibration.json', 'r', encoding='utf-8') as f:
        calibration = json.load(f)
    
    ball_diameter = 190
    camera_matrix = calibration['matrix']
    focal_distance = calibration['focal_distance']

    horizontal_radius = (ball_diameter / 2) / focal_distance * camera_matrix[0][0]
    vertical_radius = (ball_diameter / 2) / focal_distance * camera_matrix[1][1]

    area = pi * horizontal_radius * vertical_radius

    timestamp = datetime.datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d%H%M%S')

    with open(f'camera_scaling_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                "area": area,
                "distance": focal_distance
            },
            f
        )

if __name__ == '__main__':
    main()