import cv2
import numpy as np

class ChessBoardFinder:
    def __init__(self, targetWidth=800):
        self.targetWidth = targetWidth
        self.lowerBrown = np.array([10, 150, 50])
        self.upperBrown = np.array([20, 255, 200])
        self.borderPixels = 0.04769
        self.playAreaSize = 400

    def findBoard(self, image):
        resized = self.resizeImage(image)
        filtered = self.preprocessImage(resized)
        boardCorners = self.detectBoard(filtered)
        
        if boardCorners is not None:
            playCorners = self.getPlayArea(boardCorners)
            warped = self.transformPlayArea(resized, playCorners)
            return boardCorners, playCorners, filtered, warped
        return None, None, filtered, None

    def resizeImage(self, image):
        ratio = self.targetWidth / image.shape[1]
        dim = (self.targetWidth, int(image.shape[0] * ratio))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def preprocessImage(self, image):
        denoised = cv2.fastNlMeansDenoisingColored(image)
        filtered = cv2.bilateralFilter(denoised, 9, 75, 75)
        return filtered

    def detectBoard(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lowerBrown, self.upperBrown)
        
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        boardContour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(boardContour)
        return cv2.boxPoints(rect).astype(np.int32)

    def getPlayArea(self, corners):
        rect = cv2.minAreaRect(corners)
        center = rect[0]
        width, height = rect[1]
        angle = rect[2]
        
        newWidth = width * (1 - 2 * self.borderPixels)
        newHeight = height * (1 - 2 * self.borderPixels)
        
        box = cv2.boxPoints(((center[0], center[1]), (newWidth, newHeight), angle))
        return box.astype(np.int32)

    def transformPlayArea(self, image, corners):
        rect = np.zeros((4, 2), dtype="float32")
        
        s = corners.sum(axis=1)
        d = np.diff(corners, axis=1)
        
        rect[0] = corners[np.argmin(s)]  # topLeft
        rect[2] = corners[np.argmax(s)]  # bottomRight
        rect[1] = corners[np.argmin(d)]  # topRight
        rect[3] = corners[np.argmax(d)]  # bottomLeft

        dst = np.array([
            [0, 0],
            [self.playAreaSize - 1, 0],
            [self.playAreaSize - 1, self.playAreaSize - 1],
            [0, self.playAreaSize - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (self.playAreaSize, self.playAreaSize))
        squareSize = self.playAreaSize // 8
        squares = []
        for row in range(8):
            for col in range(8):
                x1 = col * squareSize
                y1 = row * squareSize
                x2 = x1 + squareSize
                y2 = y1 + squareSize
                squares.append(((x1, y1), (x2, y2)))
        
        return warped, squares

if __name__ == "__main__":
    imagePath = "images/19.jpg"
    frame = cv2.imread(imagePath)
    
    finder = ChessBoardFinder()
    boardCorners, playCorners, filtered, warpedData = finder.findBoard(frame)
    if warpedData:
        warped, squares = warpedData
    
    if boardCorners is not None:
        cv2.drawContours(filtered, [boardCorners], 0, (0, 255, 0), 2)
        cv2.drawContours(filtered, [playCorners], 0, (0, 0, 255), 2)
        warpedDisplay = warped.copy()
        for (x1, y1), (x2, y2) in squares:
            cv2.rectangle(warpedDisplay, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow('Warped', warpedDisplay)
    
    cv2.imshow('Result', filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()