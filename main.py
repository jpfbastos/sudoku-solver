import extract_puzzle
import extract_digit
import solve_puzzle
import cv2

if __name__ == '__main__':
    extract_digit.start(dataset='tmnist')

    while True:
        # initialize the camera
        cam = cv2.VideoCapture(0)

        cv2.namedWindow('test')

        img_counter = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            height, width = frame.shape[0], frame.shape[1]
            start_x, end_x = int(width/2 - height/2), int(width/2 + height/2)
            frame = frame[:, start_x:end_x]
            cv2.imshow('Press Space to take picture, and Esc to quit', frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                cv2.destroyWindow('Press Space to take picture, and Esc to quit')
                print('Escape hit, closing...')
                exit()
            elif k % 256 == 32:
                # SPACE pressed
                cv2.destroyWindow('Press Space to take picture, and Esc to quit')
                _, frame = cam.read()
                frame = frame[:, start_x:end_x]
                cv2.imshow('Is this a good image? (Press Enter to save, any other key to retake)', frame)
                k = cv2.waitKey(0)
                if k % 256 == 13:
                    img_name = 'sudoku.png'
                    cv2.imwrite(img_name, frame)
                    print('{} written!'.format(img_name))
                    break
                else:
                    cv2.destroyWindow('Is this a good image? (Press Enter to save, any other key to retake)')
                    continue

        cam.release()
        cv2.destroyAllWindows()

        puzzle, colour = extract_puzzle.read_puzzle('sudoku.png', debug=True)
        matrix = extract_puzzle.extract_array(puzzle, debug=True)
        if solve_puzzle.solve_sudoku(matrix):
            img = solve_puzzle.pretty_print(colour, matrix)
            cv2.imshow('Solved Sudoku (press Esc to quit, any other key to solve another puzzle', img)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k % 256 == 27: # ESC pressed
                print("Escape hit, closing...")
                exit()
        else:
            print("No solution was found")


