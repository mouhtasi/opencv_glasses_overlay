import pyopencv

def capture(winName):
    capture = pyopencv.VideoCapture()
    frame = pyopencv.Mat()

    capture.open(1)

    pyopencv.namedWindow(winName, pyopencv.CV_WINDOW_AUTOSIZE&1)

    if capture.isOpened():
        while True:
            capture.retrieve(frame)
            if frame.empty():
                break

            pyopencv.imshow(winName, frame)

            if pyopencv.waitKey(25) >= 0:
                break

if __name__ == '__main__':
    winName = 'PyOpenCV fun'

    capture(winName)
