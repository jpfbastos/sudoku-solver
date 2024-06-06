import cv2
import math


def find_next_empty_location(matrix, l):
    for row in range(l[0], 9):
        for col in range(l[1], 9):
            if matrix[row][col] == 0:
                l[0] = row
                l[1] = col
                return True
    return False


def used_in_row(matrix, row, num):
    if num in matrix[row]:
        return True
    return False


def used_in_column(matrix, col, num):
    for i in range(9):
        if matrix[i][col] == num:
            return True
    return False


def used_in_box(matrix, row, col, num):
    row = row - row % 3
    col = col - col % 3
    for i in range(row, row + 3):
        for j in range(col, col + 3):
            if matrix[i][j] == num:
                return True
    return False


def is_valid(matrix, row, col, num):
    return not used_in_row(matrix, row, num) and \
           not used_in_column(matrix, col, num) and \
           not used_in_box(matrix, row, col, num)


def solve_sudoku(matrix):
    loc = [0, 0]
    if not find_next_empty_location(matrix, loc):
        print(matrix)
        return True

    row = loc[0]
    col = loc[1]
    for num in range(1, 10):
        if is_valid(matrix, row, col, num):
            matrix[row][col] = num
            if solve_sudoku(matrix):
                return True
            matrix[row][col] = 0
    return False


def pretty_print(img, matrix):
    height = img.shape[0]
    width = img.shape[1]
    cell_height = height // 9
    cell_width = width // 9
    for row in range(9):
        for col in range(9):
            cv2.putText(img, str(int(matrix[row][col])), (int((col + 0.3) * cell_width),
                        int((row + 0.7) * cell_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img
