import cv2


def find_next_empty_location(matrix, loc):
    for row in range(loc[0], 9):
        for col in range(loc[1] if row == loc[0] else 0, 9):
            if matrix[row][col] == 0:
                loc = [row, col]
                return loc
    return None


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
    # Sudoku rules
    return not used_in_row(matrix, row, num) and \
           not used_in_column(matrix, col, num) and \
           not used_in_box(matrix, row, col, num)


def solve_sudoku(matrix, loc=[0, 0]):
    loc = find_next_empty_location(matrix, loc)
    if not loc:
        # All fields are completed, solution has been found
        print(matrix)
        return True

    row, col = loc
    for num in range(1, 10):
        if is_valid(matrix, row, col, num):
            # If a valid number was found, add it to the matrix and recursively feed the new matrix into the function
            matrix[row][col] = num
            if solve_sudoku(matrix, [row, col]):
                return True
            # If it does not find a solution with that number, reset it back to 0
            matrix[row][col] = 0
    return False  # Solution was not found


def pretty_print(img, matrix):
    # Divides the image in 9x9 squares
    height = img.shape[0]
    width = img.shape[1]
    cell_height = height // 9
    cell_width = width // 9
    for row in range(9):
        for col in range(9):
            # Writes text over the image
            cv2.putText(img, str(int(matrix[row][col])), (int((col + 0.3) * cell_width),
                        int((row + 0.7) * cell_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img
