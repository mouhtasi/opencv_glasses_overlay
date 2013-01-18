import pyopencv
from PIL import Image

roiLambda = 0.7
fadeoutLambda = 0.9
fadeoutLim = 50
prevROI = None
roiScaler = 2.0

def draw(frame, eyes, glasses):
    global prevROI

    imgSize = frame.size()

    if len(eyes) == 0:
        region = None
        if prevROI:
            region = pyopencv.Rect()
            roiCX = prevROI[0] + prevROI[2]/2.0
            roiCY = prevROI[1] + prevROI[3]/2.0

            newWidth = fadeoutLambda * prevROI[2]
            newHeight = fadeoutLambda * prevROI[3]/2.0

            newX = roiCX - newWidth/2.0
            newY = roiCY - newHeight/2.0

            region.x = int(round(newX))
            region.y = int(round(newY))
            region.width = int(round(newWidth))
            region.height = int(round(newHeight))

            prevROI = (newX, newY, newWidth, newHeight)
            if region.width < fadeoutLim or region.height < fadeoutLim:
                region = None
                previousROI = None

    else:
        region = eyes[0]
        if not prevROI:
            prevROI = (region.x*1.0, region.y*1.0, region.width*1.0,
                       region.height*1.0)
        else:
            newWidth = (roiLambda*prevROI[2]
                        + (1.0-roiLambda)*region.width*roiScaler)
            newHeight = (roiLambda*prevROI[3]
                         + (1.0-roiLambda)*region.height*roiScaler)
            roiCX = (roiLambda*(prevROI[0] + prevROI[2]/2.0)
                     + (1.0-roiLambda)*(region.x + region.width/2.0))
            roiCY = (roiLambda*(prevROI[1] + prevROI[3]/2.0)
                     + (1.0-roiLambda)*(region.y + region.height/2.0))

            newX = roiCX - newWidth/2.0
            newY = roiCY - newHeight/2.0

            if newX < 0:
                newX = 0
            if newY < 0:
                newY = 0
            if newX + newWidth > imgSize[0]:
                newWidth = imgSize[0] - newX
            if newY + newHeight > imgSize[1]:
                newHeight = imgSize[1] - newY

            prevROI = (newX, newY, newWidth, newHeight)

            region.x = int(round(newX))
            region.y = int(round(newY))
            region.width = int(round(newWidth))
            region.height = int(round(newHeight))

            if region.width < fadeoutLim or region.height < fadeoutLim:
                region = None
                previousROI = None

    if region:
        roiSize = region.size()
        faceROI = frame(region)

        glasses = glasses.resize((roiSize[0], roiSize[1]), Image.BICUBIC)

        glasses = pyopencv.Mat.from_pil_image(glasses)

        mask = pyopencv.Mat()
        pyopencv.cvtColor(glasses, mask, pyopencv.CV_RGB2GRAY)

        pyopencv.subtract(faceROI, faceROI, faceROI, mask)
        pyopencv.add(faceROI, glasses, faceROI, mask)

    return frame

def detect(frame, cascade, fallbackCascade):
    minPairSize = pyopencv.Size(20, 20)

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

    return eyes

if __name__ == '__main__':
    winName = 'PyOpenCV fun'

    cascade = pyopencv.CascadeClassifier()
    cascade.load("haarcascades/haarcascade_mcs_eyepair_big.xml")

    fallbackCascade = pyopencv.CascadeClassifier()
    fallbackCascade.load("haarcascades/haarcascade_mcs_eyepair_small.xml")

    glasses = Image.open('darkkamina.png')
    glasses.load()

    capture = pyopencv.VideoCapture()
    frame = pyopencv.Mat()
    capture.open(1)
    pyopencv.namedWindow(winName, pyopencv.CV_WINDOW_AUTOSIZE&1)

    if capture.isOpened():
        while True:
            capture.retrieve(frame)
            if frame.empty():
                break

            eyes = detect(frame, cascade, fallbackCascade)
            final = draw(frame, eyes, glasses)

            pyopencv.imshow(winName, final)

            if pyopencv.waitKey(25) >= 0:
                break

