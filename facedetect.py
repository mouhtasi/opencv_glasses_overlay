import pyopencv

def draw(frame, eyes):
    r = eyes[0]
    pt1 = pyopencv.Point()
    pt2 = pyopencv.Point()
    pt1.x = r.x
    pt2.x = r.x + r.width
    pt1.y = r.y
    pt2.y = r.y + r.height

    center = pyopencv.Point(int(round(r.x + r.width * 0.5)),
                            int(round(r.y + r.height * 0.5)))
    pyopencv.rectangle(frame, pt1, pt2, pyopencv.CV_RGB(255, 0, 0), 2,
                    pyopencv.CV_AA, 0)

    pyopencv.imshow(winName, frame)

def detect(frame, cascade, fallbackCascade):
    minPairSize = pyopencv.Size(50, 50)

    haarScale = 1.1
    minNeighbors = 0
    haarFlags = pyopencv.CascadeClassifier.FIND_BIGGEST_OBJECT

    grayImg = pyopencv.Mat()
    pyopencv.cvtColor(frame, grayImg, pyopencv.CV_BGR2GRAY)
    pyopencv.equalizeHist(grayImg, grayImg)

    eyes = cascade.detectMultiScale(grayImg, haarScale, minNeighbors, haarFlags,
                                    minPairSize)

    if len(eyes) == 0:
        eyes = fallbackCascade.detectMultiScale(grayImg, haarScale,
                        minNeighbors, haarFlags
                        |pyopencv.CascadeClassifier.SCALE_IMAGE, minPairSize)

    if len(eyes) != 0:
        draw(frame, eyes)

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
    fallbackCascade.load("haarcascades/haarcascade_mcs_eyepair_small.xml")

    capture(winName, cascade, fallbackCascade)
