import argparse
import os.path
import cv2
import extract_puzzle
import extract_digit
import solve_puzzle

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Sudoku Solver from Image")
    parser.add_argument('--input', '-i', type=str, default=None, help="Path to the input Sudoku image file (leave "
                                                                      "blank to use camera)")
    parser.add_argument('--dataset', '-d', type=str, choices=['mnist', 'tmnist'], default='tmnist',
                        help="Choice of dataset for the CNN model ('mnist' or 'tmnist')")
    parser.add_argument('--debug', type=str, choices=['read_puzzle', 'extract_array', 'all'], default=None,
                        help="Debug info ('read_puzzle', 'extract_array' or 'all')")

    args = parser.parse_args()
    input_path = args.input
    if input_path and not os.path.isfile(input_path):
        print("File not found")
        exit()
    dataset_choice = args.dataset
    debug_info = args.debug

    extract_digit.start(dataset_choice)

    while True:
        # Initialize the camera
        cam = cv2.VideoCapture(0)

        cv2.namedWindow('img')

        while True:
            # Continuously reads an image from the camera and displays it (basically live video)
            ret, frame = cam.read()
            if not ret:
                print('failed to grab frame')
                break
            height, width = frame.shape[0], frame.shape[1]
            # Makes the frame square (like a sudoku puzzle)
            start_x, end_x = int(width / 2 - height / 2), int(width / 2 + height / 2)
            frame = frame[:, start_x:end_x]
            cv2.imshow('Press Space to take picture, and Esc to quit', frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # esc pressed
                cv2.destroyWindow('Press Space to take picture, and Esc to quit')
                print('Escape hit, closing...')
                exit()
            elif k % 256 == 32:
                # space pressed
                cv2.destroyWindow('Press Space to take picture, and Esc to quit')
                cv2.imshow('Is this a good image? (Press Enter to save, any other key to retake)', frame)
                k = cv2.waitKey(0)
                if k % 256 == 13:
                    img_name = 'sudoku.png'
                    cv2.imwrite(img_name, frame)
                    print(f'{img_name} written!')
                    break
                else:
                    cv2.destroyWindow('Is this a good image? (Press Enter to save, any other key to retake)')

        cam.release()
        cv2.destroyAllWindows()

        puzzle, colour = extract_puzzle.read_puzzle('sudoku.png',
                                                    debug=debug_info == 'read_puzzle' or debug_info == 'all')
        matrix = extract_puzzle.extract_array(puzzle, debug=debug_info == 'extract_array' or debug_info == 'all')
        if solve_puzzle.solve_sudoku(matrix):
            img = solve_puzzle.pretty_print(colour, matrix)
            cv2.imshow('Solved Sudoku (Press Esc to quit, any other key to solve another puzzle)', img)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k % 256 == 27:  # esc pressed
                print('Escape hit, closing...')
                exit()
        else:
            print('No solution was found')
