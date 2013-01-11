import pyopencv

def detect(frame, cascade, fallbackCascade):
    minPairSize = pyopencv.Size(20, 20)

    haarScale = 1.1
    imageScale = 2
    haarFlags = pyopencv.CascadeClassifier.FIND_BIGGEST_OBJECT

    grayImg = pyopencv.Mat()
    pyopencv.cvtColor(frame, grayImg, pyopencv.CV_BGR2GRAY)
    pyopencv.equalizeHist(grayImg, grayImg)

    eyes = cascade.detectMultiScale(grayImg, haarScale, imageScale, haarFlags,
                                    minPairSize)

    if len(eyes) == 0:
        eyes = fallbackCascade.detectMultiScale(grayImg, haarScale, imageScale,
                                                haarFlags, minPairSize)

    if len(eyes) != 0:
        print 'Eye-pair detected!'

def capture(winName, cascade, fallbackCascade):
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
            detect(frame, cascade, fallbackCascade)

            if pyopencv.waitKey(25) >= 0:
                break

if __name__ == '__main__':
    winName = 'PyOpenCV fun'

    cascade = pyopencv.CascadeClassifier()
    cascade.load("haarcascades/haarcascade_mcs_eyepair_big.xml")

    fallbackCascade = pyopencv.CascadeClassifier()
    fallbackCascade.load("haarcascades/haarcascade_profileface.xml")

    capture(winName, cascade, fallbackCascade)
