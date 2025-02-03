import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

class ChessBoardFinder:
    def __init__(self, targetWidth=800, playAreaSize=400, borderPixels=0.04769):
        self.targetWidth = targetWidth
        self.playAreaSize = playAreaSize
        self.borderPixels = borderPixels
        # HSV thresholds for brown (adjust if necessary)
        self.lowerBrown = np.array([5, 130, 40])
        self.upperBrown = np.array([30, 255, 220])

    def findBoard(self, image):
        resized = self.resizeImage(image)
        filtered = self.preprocessImage(resized)
        boardCorners = self.detectBoard(filtered)
        
        if boardCorners is not None:
            warpedData = self.transformPlayArea(resized, boardCorners)
            return boardCorners, filtered, warpedData
        return None, filtered, None

    def resizeImage(self, image):
        ratio = self.targetWidth / image.shape[1]
        dim = (self.targetWidth, int(image.shape[0] * ratio))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def preprocessImage(self, image):
        # Denoise and apply a bilateral filter for edge-preserving smoothing.
        denoised = cv2.fastNlMeansDenoisingColored(image)
        return cv2.bilateralFilter(denoised, 9, 75, 75)

    def detectBoard(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lowerBrown, self.upperBrown)
        
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Choose the largest contour and approximate it to a quadrilateral.
        boardContour = max(contours, key=cv2.contourArea)
        arcLen = cv2.arcLength(boardContour, True)
        epsilon = 0.01 * arcLen
        approx = cv2.approxPolyDP(boardContour, epsilon, True)

        # Adjust epsilon until the approximation has exactly 4 points.
        while len(approx) != 4:
            epsilon += 0.002 * arcLen * (len(approx) - 4)
            approx = cv2.approxPolyDP(boardContour, epsilon, True)

        # Optionally, draw the approximated contour for visualization.
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        
        # Flatten to shape (4, 2) and return as integer coordinates.
        return approx.reshape(4, 2).astype(np.int32)

    def transformPlayArea(self, image, corners):
        # Sort the corners in order: top-left, top-right, bottom-right, bottom-left.
        rect = np.zeros((4, 2), dtype="float32")
        s = corners.sum(axis=1)
        d = np.diff(corners, axis=1)
        rect[0] = corners[np.argmin(s)]   # top-left
        rect[2] = corners[np.argmax(s)]     # bottom-right
        rect[1] = corners[np.argmin(d)]     # top-right
        rect[3] = corners[np.argmax(d)]     # bottom-left

        # Destination points for a square warped board.
        dst = np.array([
            [0, 0],
            [self.playAreaSize - 1, 0],
            [self.playAreaSize - 1, self.playAreaSize - 1],
            [0, self.playAreaSize - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (self.playAreaSize, self.playAreaSize))

        # Crop the warped image inward by a margin determined by borderPixels.
        margin = int(self.borderPixels * self.playAreaSize)
        cropped = warped[margin:self.playAreaSize - margin, margin:self.playAreaSize - margin]

        # Divide the cropped play area into an 8x8 grid.
        croppedSize = self.playAreaSize - 2 * margin
        squareSize = croppedSize // 8
        squares = []
        for row in range(8):
            for col in range(8):
                x1 = col * squareSize
                y1 = row * squareSize
                x2 = x1 + squareSize
                y2 = y1 + squareSize
                squares.append(((x1, y1), (x2, y2)))

        return cropped, squares

def main():
    # Initialize the Raspberry Pi camera.
    camera = PiCamera()
    camera.framerate = 32
    rawCapture = PiRGBArray(camera)

    # Allow the camera to warm up.
    time.sleep(0.1)
    
    finder = ChessBoardFinder()
    
    print("Starting video stream. Press 'q' to exit.")
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        
        boardCorners, filtered, warpedData = finder.findBoard(image)
        if boardCorners is not None and warpedData is not None:
            warped, squares = warpedData
            # Draw detected board contour on the filtered image.
            cv2.drawContours(filtered, [boardCorners], 0, (0, 255, 0), 2)
            
            # Draw individual chess squares on the warped (cropped) play area.
            warpedDisplay = warped.copy()
            for (x1, y1), (x2, y2) in squares:
                cv2.rectangle(warpedDisplay, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow("Warped (Cropped Play Area)", warpedDisplay)
        
        cv2.imshow("Result", filtered)
        
        # Clear the stream in preparation for the next frame.
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
