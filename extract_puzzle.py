from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import extract_digit


def read_puzzle(img, debug=False):
    img = cv2.imread(img)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img_grey, (9, 9), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    contours = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzle_cnt = None
    # loop over the contours
    for c in contours_sorted:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzle_cnt = approx
            break

    cv2.drawContours(img, [puzzle_cnt], -1, (0, 255, 0), 2)
    puzzle_binary = four_point_transform(binary, puzzle_cnt.reshape(4, 2))
    puzzle_binary = cv2.resize(puzzle_binary, (450, 450))
    puzzle_colour = four_point_transform(img, puzzle_cnt.reshape(4, 2))
    puzzle_colour = cv2.resize(puzzle_colour, (450, 450))

    if debug:
        cv2.imshow("Blurred", blurred)
        cv2.waitKey(0)
        binary1 = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        binary1 = cv2.drawContours(binary1, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Binary contours", binary1)
        cv2.waitKey(0)
        cv2.drawContours(img, [puzzle_cnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", img)
        cv2.waitKey(0)
        cv2.imshow("Test", puzzle_binary)
        cv2.waitKey(0)

    return puzzle_binary, puzzle_colour


def extract_array(puzzle, debug=False):
    squares = np.zeros([9, 9])
    height = puzzle.shape[0]
    width = puzzle.shape[1]
    cell_height = height // 9
    cell_width = width // 9
    for row in range(9):
        for col in range(9):
            digit = puzzle[row * cell_height:(row + 1) * cell_height, col * cell_width:(col + 1) * cell_width]
            _, thresh = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            digit = clear_border(thresh, round((cell_height + cell_width) / 2 * 0.03))
            if np.average(digit) > 5:
                coords = cv2.findNonZero(digit)
                x, y, w, h = cv2.boundingRect(coords)
                # Crop the digit from the image
                cropped_digit = digit[y:y + h, x:x + w]
                # Resize the digit to make it larger, keeping aspect ratio
                new_size = (int(40/h * w), 40)
                resized_digit = cv2.resize(cropped_digit, new_size, interpolation=cv2.INTER_AREA)
                digit = np.zeros((50, 50), dtype=np.uint8)
                # Calculate the position to center the resized digit in the 45x45 image
                x_offset = (50 - resized_digit.shape[1]) // 2
                y_offset = (50 - resized_digit.shape[0]) // 2
                # Place the resized digit into the center of the new 45x45 image
                digit[y_offset:y_offset + resized_digit.shape[0], x_offset:x_offset+resized_digit.shape[1]] = resized_digit
                if debug:
                    cv2.imshow("Digit", digit)
                    cv2.waitKey(0)
                digit = cv2.resize(digit, (28, 28))  # For plt.imshow
                digit = digit.reshape(1, 28, 28, 1)  # For input to model.predict_classes
                number = extract_digit.extract_number(digit)
                squares[row][col] = number
    if debug:
        print(squares)

    return squares