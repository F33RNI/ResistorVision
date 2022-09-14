import math
import os

import cv2
import numpy as np

# Camera settings
CAMERA_ID = 2
CAMERA_DIRECT_SHOW = True
CAMERA_EXPOSURE = -7
CAMERA_FOCUS = 40
CAMERA_AUTO_WHITE_BALANCE = False
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Preview window size
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720

# Number of stripes in resistor
RESISTOR_STRIPES = 5

# 5-Band tolerance color index (default = 1 (brown))
TOLERANCE_5BAND_COLOR_INDEX = 1

# Min and maximum area of body segment in pixels
RESISTOR_BODY_AREA_MIN = 10000
RESISTOR_BODY_AREA_MAX = 100000

# Debug
SHOW_CANNY = False
SHOW_BODY_MASK = False

# Default values for sliders
RESISTOR_BODY_HSV_RANGES_DEFAULT = [5, 100, 100]

resistor_body_hsv = [0, 0, 0]
resistor_body_hsv_ranges = [0, 0, 0]

resistor_stripes_colors_bgr = [[0] * 3 for _ in range(RESISTOR_STRIPES)]
resistor_stripes_demo_top = [[0] * 3 for _ in range(RESISTOR_STRIPES)]
resistor_stripes_text_top = '0R'


def canny_thresh_lower_change(value):
    global canny_thresh_lower
    canny_thresh_lower = value


def canny_thresh_upper_change(value):
    global canny_thresh_upper
    canny_thresh_upper = value


def resistor_body_h_range_change(value):
    global resistor_body_hsv_ranges
    resistor_body_hsv_ranges[0] = value


def resistor_body_s_range_change(value):
    global resistor_body_hsv_ranges
    resistor_body_hsv_ranges[1] = value


def resistor_body_v_range_change(value):
    global resistor_body_hsv_ranges
    resistor_body_hsv_ranges[2] = value


def color_trackbar_callback(color_index):
    global calibration_current_color_index
    calibration_current_color_index = color_index


# All possible resistances
RESISTORS_TABLE = [1, 1.2, 1.5, 1.8, 2, 2.2, 2.4, 2.7, 3, 3.3, 3.6, 3.9,
                   4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1, 10, 12, 15,
                   18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51,
                   56, 62, 68, 75, 82, 91, 100, 120, 150, 180, 200, 220,
                   240, 270, 300, 330, 360, 390, 430, 470, 510, 560, 620, 680, 750, 820, 910,
                   1e3, 1.2e3, 1.5e3, 1.8e3, 2e3, 2.2e3, 2.4e3, 2.7e3, 3e3, 3.3e3, 3.6e3, 3.9e3,
                   4.3e3, 4.7e3, 5.1e3, 5.6e3, 6.2e3, 6.8e3, 7.5e3, 8.2e3, 9.1e3, 10e3, 12e3, 15e3,
                   18e3, 20e3, 22e3, 24e3, 27e3, 30e3, 33e3, 36e3, 39e3, 43e3, 47e3, 51e3,
                   56e3, 62e3, 68e3, 75e3, 82e3, 91e3, 100e3, 120e3, 150e3, 180e3, 200e3, 220e3,
                   240e3, 270e3, 300e3, 330e3, 360e3, 390e3, 430e3, 470e3, 510e3, 560e3, 620e3, 680e3,
                   750e3, 820e3, 910e3,
                   1e6, 1.5e6, 2e6, 3e6]

COLOR_NAMES = ['BLACK', 'BROWN', 'RED', 'ORANGE', 'YELLOW', 'GREEN', 'BLUE', 'VIOLET', 'GRAY', 'WHITE',
               'GOLD', 'SILVER']

COLORS_DEMO_RGB = [[0, 0, 0], [140, 78, 45], [255, 0, 0], [255, 127, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255],
                   [170, 0, 255], [127, 127, 127], [255, 255, 255], [199, 180, 74], [181, 181, 181]]

COLORS_FILE = 'colors_rgb.csv'
COLORS_FILE_SEPARATOR = ','

colors = [[0] * 3 for _ in range(len(COLOR_NAMES))]
colors_calibration = [[0] * 3 for _ in range(len(COLOR_NAMES))]
colors_calibration_counters = [0] * len(COLOR_NAMES)

calibration_current_color_index = 0


def load_colors_file():
    if os.path.exists(COLORS_FILE):
        colors_counter = 0
        colors_file = open(COLORS_FILE, 'r')
        colors_file_lines = colors_file.readlines()
        for line in colors_file_lines:
            line = line.replace('\n', '').replace('\r', '').replace(' ', '')
            parts_of_line = line.split(COLORS_FILE_SEPARATOR)
            if len(parts_of_line) == 3:
                colors[colors_counter][0] = int(parts_of_line[0])
                colors[colors_counter][1] = int(parts_of_line[1])
                colors[colors_counter][2] = int(parts_of_line[2])
                colors_counter += 1
            else:
                colors_file.close()
                return False

        colors_file.close()
        return True
    else:
        return False


def resize_keep_ratio(source_image, target_width, target_height, interpolation=cv2.INTER_AREA):
    """
    Resizes image and keeps aspect ratio (fills background with black)
    """
    border_v = 0
    border_h = 0
    if (target_height / target_width) >= (source_image.shape[0] / source_image.shape[1]):
        border_v = int((((target_height / target_width) * source_image.shape[1]) - source_image.shape[0]) / 2)
    else:
        border_h = int((((target_width / target_height) * source_image.shape[0]) - source_image.shape[1]) / 2)
    output_image = cv2.copyMakeBorder(source_image, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
    return cv2.resize(output_image, (target_width, target_height), interpolation)


def color_indexes_to_resistance(color_indexes):
    reversed_ = False
    resistance = 0

    if len(color_indexes) == 3:
        # 3-band resistor, start from gold or silver -> reversed resistor
        if color_indexes[0] == 10 or color_indexes[0] == 11:
            reversed_ = True
            base = color_indexes[2] * 10 + color_indexes[1]
            resistance = int(base * math.pow(10, color_indexes[0]))

        # 3-band not reversed resistor
        else:
            base = color_indexes[0] * 10 + color_indexes[1]
            resistance = int(base * math.pow(10, color_indexes[2]))

    elif len(color_indexes) == 4:
        # 4-band resistor, start from gold or silver -> reversed resistor
        if color_indexes[0] == 10 or color_indexes[0] == 11:
            reversed_ = True
            base = color_indexes[3] * 10 + color_indexes[2]
            resistance = int(base * math.pow(10, color_indexes[1]))

        # 4-band not reversed resistor
        else:
            base = color_indexes[0] * 10 + color_indexes[1]
            resistance = int(base * math.pow(10, color_indexes[2]))

    elif len(color_indexes) == 5:
        # 5-band resistor, tolerance color at start or gold or silver at second -> reversed resistor
        if (color_indexes[0] == TOLERANCE_5BAND_COLOR_INDEX and color_indexes[-1] != TOLERANCE_5BAND_COLOR_INDEX) \
                or color_indexes[1] == 10 or color_indexes[1] == 11:
            reversed_ = True
            base = color_indexes[4] * 100 + color_indexes[3] * 10 + color_indexes[2]

            # 5-band resistor, gold multiplier -> 0.1
            if color_indexes[1] == 10:
                resistance = base * 0.1

            # 5-band resistor, silver multiplier -> 0.01
            elif color_indexes[1] == 11:
                resistance = base * 0.01

            # 5-band resistor, normal multiplier
            else:
                resistance = int(base * math.pow(10, color_indexes[1]))

        # 5-band not reversed resistor
        else:
            base = color_indexes[0] * 100 + color_indexes[1] * 10 + + color_indexes[2]

            # 5-band resistor, gold multiplier -> 0.1
            if color_indexes[3] == 10:
                resistance = base * 0.1

            # 5-band resistor, silver multiplier -> 0.01
            elif color_indexes[3] == 11:
                resistance = base * 0.01

            # 5-band resistor, normal multiplier
            else:
                resistance = int(base * math.pow(10, color_indexes[3]))

    if reversed_:
        # Starts from gold or silver -> wrong resistor
        if color_indexes[-1] == 10 or color_indexes[-1] == 11:
            return False, reversed_, resistance
    else:
        # Starts from gold or silver -> wrong resistor
        if color_indexes[0] == 10 or color_indexes[0] == 11:
            return False, reversed_, resistance

    # Check if resistance is in table
    success = resistance in RESISTORS_TABLE

    return success, reversed_, resistance


def resistance_to_text(resistance):
    # Ohms
    if resistance < 1000:
        fraction = round((resistance % 1) * 10)
        return str(int(resistance)) + 'R' + (str(fraction) if fraction > 0 else '')

    # KiloOhms
    elif resistance < 1000000:
        resistance /= 1000
        fraction = round((resistance % 1) * 10)
        return str(int(resistance)) + 'K' + (str(fraction) if fraction > 0 else '')

    # MegaOhms
    else:
        resistance /= 1000000
        fraction = round((resistance % 1) * 10)
        return str(int(resistance)) + 'M' + (str(fraction) if fraction > 0 else '')


if __name__ == '__main__':
    # Initialize camera
    if CAMERA_DIRECT_SHOW:
        capture = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(CAMERA_ID)
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    capture.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    capture.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)
    capture.set(cv2.CAP_PROP_AUTO_WB, 1 if CAMERA_AUTO_WHITE_BALANCE else 0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    capture.read()

    trackbars_created = False

    resistor_body_hsv_set = False

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))

    colors_loaded = load_colors_file()

    calibration_trackbars_created = [False] * RESISTOR_STRIPES

    while True:
        # Set camera focus
        capture.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)

        # Capture frame from camera
        ret, frame = capture.read()

        result_frame = frame.copy()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hue_lower = max(0, resistor_body_hsv[0] - resistor_body_hsv_ranges[0])
        hue_upper = min(179, resistor_body_hsv[0] + resistor_body_hsv_ranges[0])
        saturation_lower = max(0, resistor_body_hsv[1] - resistor_body_hsv_ranges[1])
        saturation_upper = min(255, resistor_body_hsv[1] + resistor_body_hsv_ranges[1])
        value_lower = max(0, resistor_body_hsv[2] - resistor_body_hsv_ranges[2])
        value_upper = min(255, resistor_body_hsv[2] + resistor_body_hsv_ranges[2])

        frame_threshold = cv2.inRange(frame_hsv, (hue_lower, saturation_lower, value_lower),
                                      (hue_upper, saturation_upper, value_upper))

        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _, threshold = cv2.threshold(gray, CANNY_THRESH_UPPER, 255, cv2.THRESH_BINARY)

        canny = cv2.Canny(frame, canny_thresh_lower, canny_thresh_upper)

        canny = cv2.dilate(canny, kernel)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Show canny filter
        if SHOW_CANNY:
            result_frame[:, :, 0] = cv2.bitwise_and(result_frame[:, :, 0], cv2.bitwise_not(canny))
            result_frame[:, :, 1] = cv2.bitwise_and(result_frame[:, :, 1], cv2.bitwise_not(canny))
            result_frame[:, :, 2] = cv2.bitwise_and(result_frame[:, :, 2], cv2.bitwise_not(canny))

        if contours is not None and len(contours) > 0:
            largest_contour = contours[0]
            largest_contour_area = 0

            for i in range(len(contours)):
                contour = contours[i]
                area = cv2.contourArea(contour)
                if area > largest_contour_area:
                    largest_contour_area = area
                    largest_contour = contour

            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            cv2.drawContours(result_frame, [largest_contour], -1, (0, 0, 0), 1)
        """

        frame_threshold = cv2.erode(frame_threshold, kernel_erode)
        frame_threshold = cv2.dilate(frame_threshold, kernel_dilate)

        if SHOW_BODY_MASK:
            result_frame[:, :, 0] = cv2.bitwise_and(result_frame[:, :, 0], cv2.bitwise_not(frame_threshold))
            result_frame[:, :, 1] = cv2.bitwise_and(result_frame[:, :, 1], cv2.bitwise_not(frame_threshold))
            result_frame[:, :, 2] = cv2.bitwise_and(result_frame[:, :, 2], cv2.bitwise_not(frame_threshold))

        contours, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours is not None and len(contours) > 0:
            contours = np.array(contours)
            contours_filtered = []

            for i in range(len(contours)):
                contour = contours[i]
                area = cv2.contourArea(contour)

                if RESISTOR_BODY_AREA_MIN <= area <= RESISTOR_BODY_AREA_MAX:
                    contours_filtered.append(contour)

            contours_filtered = np.array(contours_filtered)
            cv2.drawContours(result_frame, contours_filtered, -1, (0, 0, 0))

            if len(contours_filtered) == RESISTOR_STRIPES + 1:
                x_max = 0
                y_max = 0
                x_min = CAMERA_WIDTH * 10
                y_min = CAMERA_HEIGHT * 10

                contours_filtered_rectangles = []

                for contour in contours_filtered:
                    x, y, w, h = cv2.boundingRect(contour)

                    contours_filtered_rectangles.append([[[x, y]],
                                                         [[x + w, y]],
                                                         [[x + w, y + h]],
                                                         [[x, y + h]]])

                    for point in contour:
                        contour_x = point[0][0]
                        contour_y = point[0][1]
                        if contour_x > x_max:
                            x_max = contour_x
                        if contour_x < x_min:
                            x_min = contour_x
                        if contour_y > y_max:
                            y_max = contour_y
                        if contour_y < y_min:
                            y_min = contour_y

                contours_filtered_rectangles = np.array(contours_filtered_rectangles)
                boundingBoxes = [cv2.boundingRect(c) for c in contours_filtered_rectangles]
                (contours_filtered_rectangles, boundingBoxes) = zip(
                    *sorted(zip(contours_filtered_rectangles, boundingBoxes),
                            key=lambda b: b[1][0]))

                resistor_contour = [[[x_min, y_min]],
                                    [[x_max, y_min]],
                                    [[x_max, y_max]],
                                    [[x_min, y_max]]]

                cv2.drawContours(result_frame, np.array([resistor_contour]), -1, (0, 255, 0))
                cv2.drawContours(result_frame, contours_filtered_rectangles, -1, (0, 255, 0))

                stripes_contours = []

                for i in range(RESISTOR_STRIPES):
                    x_tl = contours_filtered_rectangles[i][1][0][0]
                    y_tl = contours_filtered_rectangles[i][1][0][1]
                    x_br = contours_filtered_rectangles[i + 1][3][0][0]
                    y_br = contours_filtered_rectangles[i + 1][3][0][1]

                    stripes_contours.append([[[x_tl, y_tl]],
                                             [[x_br, y_tl]],
                                             [[x_br, y_br]],
                                             [[x_tl, y_br]]])

                stripes_contours = np.array(stripes_contours)

                for i in range(RESISTOR_STRIPES):
                    stripe_contour_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype='uint8')
                    cv2.drawContours(stripe_contour_mask, stripes_contours, i, 255, -1)
                    mean = np.array(cv2.mean(frame, mask=stripe_contour_mask)).astype('uint8')
                    mean_hsv = cv2.cvtColor(np.array([[mean]]), cv2.COLOR_BGR2HSV)[0][0]
                    resistor_stripes_colors_bgr[i] = [int(mean[0]), int(mean[1]), int(mean[2])]

                    cv2.drawContours(result_frame, stripes_contours, i, (int(mean[0]), int(mean[1]), int(mean[2])), 2)

                    if not colors_loaded:
                        for stripe_index in range(RESISTOR_STRIPES):
                            stripe_color = resistor_stripes_colors_bgr[stripe_index]

                            stripe_frame = np.ones((len(COLOR_NAMES) * 20 + 20, 250, 3), dtype='uint8')
                            stripe_frame[:, :, 0] *= stripe_color[0]
                            stripe_frame[:, :, 1] *= stripe_color[1]
                            stripe_frame[:, :, 2] *= stripe_color[2]

                            for demo_color_index in range(len(COLOR_NAMES)):
                                cv2.putText(stripe_frame, str(demo_color_index) + " - " + COLOR_NAMES[demo_color_index],
                                            (0, demo_color_index * 20 + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 1,
                                            (COLORS_DEMO_RGB[demo_color_index][2],
                                             COLORS_DEMO_RGB[demo_color_index][1],
                                             COLORS_DEMO_RGB[demo_color_index][0]), 1)

                            cv2.imshow('Stripe ' + str(stripe_index + 1), stripe_frame)
                            if not calibration_trackbars_created[stripe_index]:
                                cv2.createTrackbar('Color', 'Stripe ' + str(stripe_index + 1), 0, len(COLOR_NAMES) - 1,
                                                   color_trackbar_callback)
                                calibration_trackbars_created[stripe_index] = True

                    # Colors loaded from file
                    else:

                        distances_per_stripes = []

                        for stripe_index in range(RESISTOR_STRIPES):
                            stripe_color = resistor_stripes_colors_bgr[stripe_index]

                            distances_per_stripe = []

                            for reference_color_index in range(len(colors)):
                                reference_color = colors[reference_color_index]

                                distance = abs(reference_color[0] - stripe_color[0]) \
                                           + abs(reference_color[1] - stripe_color[1]) \
                                           + abs(reference_color[2] - stripe_color[2])

                                distances_per_stripe.append([reference_color_index, distance])

                            distances_per_stripes.append(distances_per_stripe)

                        distances_per_stripes_sorted = []

                        for distances_per_stripe_index in range(len(distances_per_stripes)):
                            distances_per_stripe = distances_per_stripes[distances_per_stripe_index]
                            distances_per_stripe.sort(key=lambda element: element[1])
                            distances_per_stripes_sorted.append(distances_per_stripe)

                        # distances_per_stripes.sort(key=lambda distances_per_stripes:distances_per_stripes[1])

                        indexes_and_avg_distances_all = []

                        for depth_counter in range(2):
                            for stripe_index in range(RESISTOR_STRIPES + 1):
                                distance_avg = 0
                                indexes_and_distances = []
                                for stripe_index_ in range(RESISTOR_STRIPES):
                                    if stripe_index == stripe_index_:
                                        distance_avg += \
                                            distances_per_stripes_sorted[stripe_index_][depth_counter + 1][1]
                                        indexes_and_distances.append(
                                            distances_per_stripes_sorted[stripe_index_][depth_counter + 1])
                                    else:
                                        distance_avg += distances_per_stripes_sorted[stripe_index_][depth_counter][1]
                                        indexes_and_distances.append(
                                            distances_per_stripes_sorted[stripe_index_][depth_counter])

                                distance_avg /= RESISTOR_STRIPES
                                indexes_and_avg_distances_all.append([indexes_and_distances, distance_avg])

                        indexes_and_avg_distances_all.sort(key=lambda element: element[1])

                        color_indexes_variants_final_not_filtered = []

                        for indexes_and_avg_distances in indexes_and_avg_distances_all:
                            indexes = []
                            for index_and_distance in indexes_and_avg_distances[0]:
                                indexes.append(index_and_distance[0])
                            color_indexes_variants_final_not_filtered.append(indexes)

                        color_indexes_variants_final = []

                        for element in color_indexes_variants_final_not_filtered:
                            element_reversed = element.copy()
                            element_reversed.reverse()

                            # Check if sequence or reversed sequence is already in list
                            if element not in color_indexes_variants_final \
                                    and element_reversed not in color_indexes_variants_final:
                                color_indexes_variants_final.append(element)

                        resistances_final = []
                        for color_indexes_variant_final in color_indexes_variants_final:
                            # Convert colors to resistance
                            success, reversed_, resistance = color_indexes_to_resistance(color_indexes_variant_final)

                            # Check if conversion is successfully
                            if success:
                                # Convert resistance to text code
                                text = resistance_to_text(resistance)

                                # Check if code is not in list and list size < 4 (max 4 elements)
                                if text not in resistances_final and len(resistances_final) < 4:
                                    # Append colors and text to the list
                                    resistances_final.append([color_indexes_variant_final, text])

                        if len(resistances_final) > 0:
                            color_indexes_top = resistances_final[0][0]
                            for color_index_top in range(len(color_indexes_top)):
                                resistor_stripes_demo_top[color_index_top][0] = \
                                COLORS_DEMO_RGB[color_indexes_top[color_index_top]][2]
                                resistor_stripes_demo_top[color_index_top][1] = \
                                COLORS_DEMO_RGB[color_indexes_top[color_index_top]][1]
                                resistor_stripes_demo_top[color_index_top][2] = \
                                COLORS_DEMO_RGB[color_indexes_top[color_index_top]][0]
                            resistor_stripes_text_top = resistances_final[0][1]

                        color_code_start_x = resistor_contour[3][0][0]
                        color_code_start_y = resistor_contour[3][0][1] + 20

                        # List all colors and resistances
                        for resistance_final_index in range(len(resistances_final)):
                            # Get colors and text
                            resistance_final_colors = resistances_final[resistance_final_index][0]
                            resistance_final_text = resistances_final[resistance_final_index][1]

                            # List all colors
                            for resistance_final_color_index in range(len(resistance_final_colors)):
                                # Get demo RGB color
                                resistance_final_color_rgb = \
                                    COLORS_DEMO_RGB[resistance_final_colors[resistance_final_color_index]]

                                # Convert it to BGR
                                resistance_final_color_bgr = [resistance_final_color_rgb[2]
                                    , resistance_final_color_rgb[1], resistance_final_color_rgb[0]]

                                # Calculate color rectangle
                                color_rect_tl_x = color_code_start_x + resistance_final_color_index * 40
                                color_rect_tl_y = color_code_start_y + resistance_final_index * 60
                                color_rect_br_x = color_code_start_x + resistance_final_color_index * 40 + 35
                                color_rect_br_y = color_code_start_y + resistance_final_index * 60 + 40

                                # Add distance between first color code and all others
                                if resistance_final_index > 0:
                                    color_rect_tl_y += 20
                                    color_rect_br_y += 20

                                # Draw color rectangle
                                cv2.rectangle(result_frame, (color_rect_tl_x, color_rect_tl_y),
                                              (color_rect_br_x, color_rect_br_y), resistance_final_color_bgr, -1)

                            # Calculate text position
                            text_x = color_code_start_x + (len(resistance_final_colors) + 1) * 40
                            text_y = color_code_start_y + resistance_final_index * 60 + 30

                            # Add distance between first color code and all others
                            if resistance_final_index > 0:
                                text_y += 20

                            # Print text white on black
                            cv2.putText(result_frame, resistance_final_text, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                            cv2.putText(result_frame, resistance_final_text, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)



        # Draw stripes on top of the screen
        for i in range(RESISTOR_STRIPES):
            cv2.rectangle(result_frame, (i * 40, 0), (i * 40 + 35, 100), resistor_stripes_colors_bgr[i], -1)
            cv2.rectangle(result_frame, (i * 40, 105), (i * 40 + 35, 205), resistor_stripes_demo_top[i], -1)

        # Draw text on top of the screen
        cv2.putText(result_frame, resistor_stripes_text_top, ((RESISTOR_STRIPES + 1) * 40, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 15)
        cv2.putText(result_frame, resistor_stripes_text_top, ((RESISTOR_STRIPES + 1) * 40, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 255, 255), 11)

        # No resistor body color -> request ROI
        if not resistor_body_hsv_set:
            frame_to_roi = frame.copy()
            cv2.putText(frame_to_roi, 'Select BODY COLOR', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(frame_to_roi, 'and press ENTER', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(frame_to_roi, 'Or press ENTER to skip frame', (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 2)
            roi = cv2.selectROI('Resistor body color', frame_to_roi)

            if roi != (0, 0, 0, 0):
                cv2.destroyAllWindows()

                roi_cropped = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

                mean = np.array(cv2.mean(roi_cropped)).astype('uint8')
                print('Resistor body color (RGB): ', [int(mean[2]), int(mean[1]), int(mean[0])])

                mean_hsv = cv2.cvtColor(np.array([[mean]]), cv2.COLOR_BGR2HSV)[0][0]
                resistor_body_hsv = mean_hsv.tolist()

                print('Resistor body color (HSV): ', mean_hsv)

                resistor_body_hsv_set = True

        # Resistor body color is set -> display the resulting frame
        if resistor_body_hsv_set:
            cv2.imshow('ResistorVision', resize_keep_ratio(result_frame, PREVIEW_WIDTH, PREVIEW_HEIGHT))

            if not trackbars_created:
                cv2.createTrackbar('Body H+-', 'ResistorVision', 0, 50, resistor_body_h_range_change)
                cv2.setTrackbarPos('Body H+-', 'ResistorVision', RESISTOR_BODY_HSV_RANGES_DEFAULT[0])
                cv2.createTrackbar('Body S+-', 'ResistorVision', 0, 200, resistor_body_s_range_change)
                cv2.setTrackbarPos('Body S+-', 'ResistorVision', RESISTOR_BODY_HSV_RANGES_DEFAULT[1])
                cv2.createTrackbar('Body V+-', 'ResistorVision', 0, 200, resistor_body_v_range_change)
                cv2.setTrackbarPos('Body V+-', 'ResistorVision', RESISTOR_BODY_HSV_RANGES_DEFAULT[2])
                trackbars_created = True

        key = cv2.waitKey(1) & 0xFF

        for i in range(RESISTOR_STRIPES):
            if key == ord(str(i + 1)):
                print('Stripe n', i + 1, 'is', COLOR_NAMES[calibration_current_color_index])
                if colors_calibration_counters[calibration_current_color_index] == 0:
                    colors_calibration[calibration_current_color_index] = resistor_stripes_colors_bgr[i]
                    colors_calibration_counters[calibration_current_color_index] = 1
                else:
                    colors_calibration[calibration_current_color_index][0] += resistor_stripes_colors_bgr[i][0]
                    colors_calibration[calibration_current_color_index][1] += resistor_stripes_colors_bgr[i][1]
                    colors_calibration[calibration_current_color_index][2] += resistor_stripes_colors_bgr[i][2]
                    colors_calibration_counters[calibration_current_color_index] += 1

        # Finish calibration
        if key == ord('c'):
            all_colors_calibrated = True
            need_to_calibrate = ''

            for i in range(len(colors_calibration)):
                calibrated_n = colors_calibration_counters[i]
                if calibrated_n < 1:
                    all_colors_calibrated = False
                    need_to_calibrate += COLOR_NAMES[i] + ' '

            if all_colors_calibrated:
                with open(COLORS_FILE, 'w+') as colors_file:
                    for i in range(len(colors_calibration)):
                        color = colors_calibration[i]
                        calibrated_n = colors_calibration_counters[i]
                        color_b = str(int(color[0] / calibrated_n))
                        color_g = str(int(color[1] / calibrated_n))
                        color_r = str(int(color[2] / calibrated_n))

                        colors_file.write(color_b + COLORS_FILE_SEPARATOR
                                          + color_g + COLORS_FILE_SEPARATOR
                                          + color_r + '\n')

                    colors_file.close()
                    print('Calibration done!')
                    capture.release()
                    cv2.destroyAllWindows()
                    exit(0)
                    break

            else:
                print('Not all colors are calibrated!')
                print('Please calibrate these colors: ' + need_to_calibrate)

        # Press Q to quit
        if key == ord('q'):
            break

    # Close camera
    capture.release()
    cv2.destroyAllWindows()
