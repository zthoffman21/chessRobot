import cv2
import numpy as np

def detectPiecesInSquares(playingArea: np.ndarray) -> list[list[int]]:
    """
    Leverages the fact that we know which square are supposed to be light and dark in addition to variance to detect pieces and their color in each square.

    Args:
        playingArea (np.ndarray): The processed playing area transformed into a 400x400 size. 

    Returns:
        list[list[int]]: A 2d array showing the current board with 0 = empty square, 1 = black piece, and 2 = white piece.
    """
    board = [[0,0,0,0,0,0,0,0] for x in range(8)]

    squareOutput = cv2.cvtColor(playingArea, cv2.COLOR_GRAY2BGR)
    squareSize = playingArea.shape[0] // 8
    
    # Define center region size
    centerSize = int(squareSize * 0.4)
    offset = (squareSize - centerSize) // 2
    
    for row in range(8):
        for col in range(8):
            # Calculate square regions
            x1, y1 = col * squareSize, row * squareSize
            centerX1, centerY1 = x1 + offset, y1 + offset
            centerX2, centerY2 = centerX1 + centerSize, centerY1 + centerSize
            
            # Extract center region and get stats
            centerRegion = playingArea[centerY1:centerY2, centerX1:centerX2]
            minIntensity = np.min(centerRegion)
            maxIntensity = np.max(centerRegion)
            variance = np.var(centerRegion)
            
            # Check for pieces based on square color
            isWhiteSquare = (row + col) % 2 == 0
            isWhitePiece = False
            hasPiece = False
            
            if isWhiteSquare:
                # Check for pieces on white squares
                if minIntensity < 50:  # Black piece on white
                    hasPiece = True
                    board[row][col] = 1
                elif variance > 300 or maxIntensity > 220:  # White piece on white
                    hasPiece = True
                    isWhitePiece = True
                    board[row][col] = 2
            else:
                # Check for pieces on black squares
                if maxIntensity > 160:  # White piece on black
                    hasPiece = True
                    isWhitePiece = True
                    board[row][col] = 2
                elif variance > 90 or minIntensity < 30:  # Black piece on black
                    hasPiece = True
                    board[row][col] = 1
            
            # Draw visualization
            if hasPiece:
                if isWhitePiece:
                    overlay = squareOutput.copy()
                    cv2.rectangle(overlay, (x1, y1), (x1 + squareSize, y1 + squareSize), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, squareOutput, 0.7, 0, squareOutput)
                else:
                    overlay = squareOutput.copy()
                    cv2.rectangle(overlay, (x1, y1), (x1 + squareSize, y1 + squareSize), (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, squareOutput, 0.7, 0, squareOutput)
            
            # Draw square borders and center region
            cv2.rectangle(squareOutput, (x1, y1), (x1 + squareSize, y1 + squareSize), (0, 255, 0), 1)
            cv2.rectangle(squareOutput, (centerX1, centerY1), (centerX2, centerY2), (0, 0, 255), 1)
            
            # Add debug text
            cv2.putText(squareOutput, f'{row},{col}', (x1+10, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(squareOutput, f'min:{minIntensity:.0f}', (x1+10, y1+35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.putText(squareOutput, f'max:{maxIntensity:.0f}', (x1+10, y1+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            

    cv2.imshow('Square Detection', squareOutput)
    return board

def sortCorners(corners):
    # Print corner format for debugging
    print("Corner shape:", corners.shape)
    print("Corners:", corners)
    
    # Reshape corners to simple 2D array if needed
    if len(corners.shape) > 2:
        corners = corners.reshape(4, 2)
        
    # Sort corners by Y first (top vs bottom)
    sorted_by_y = sorted(corners, key=lambda p: p[1])
    
    # Split into top and bottom pairs
    top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
    bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])
    
    return np.array([top_two[0], top_two[1], 
                    bottom_two[1], bottom_two[0]], dtype=np.float32)

def detectChessboardBoundary(imagePath: str, windowSize=800) -> list[list[int]]:
    """
    Uses some fundamental image processing to detect where in the image the chess board is.

    Args:
        imagePath (str): The path to the file containing an image of the chess board
        windowSize (int, optional): Optional argument to control the size of the output window. Defaults to 800.

    Returns:
        list[list[int]]: A 2d array showing the current board with 0 = empty square, 1 = black piece, and 2 = white piece.
    """
    # Read and preprocess image
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize if needed
    h, w = gray.shape
    if h > windowSize or w > windowSize:
        scale = windowSize / max(h, w)
        newW, newH = int(w * scale), int(h * scale)
        img = cv2.resize(img, (newW, newH))
        gray = cv2.resize(gray, (newW, newH))

    # Clean up image
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.dilate(gray, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Process image for contour detection
    thresh = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 35, 3)
    
    kernel = np.ones((4,4), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    # Find chessboard boundaries
    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    output = img.copy()
    
    outerBoundary = None
    innerBoundary = None
    largestArea = secondLargestArea = 0

    # Find two largest square-like contours
    for contour in contours:
        area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if 0.8 <= float(w)/h <= 1.2:
                if area > largestArea:
                    secondLargestArea = largestArea
                    innerBoundary = outerBoundary
                    largestArea = area
                    outerBoundary = approx
                elif area > secondLargestArea:
                    secondLargestArea = area
                    innerBoundary = approx

        # Process detected boundaries
    if outerBoundary is not None and innerBoundary is not None:
        # Draw boundaries
        x, y, w, h = cv2.boundingRect(outerBoundary)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.drawContours(output, [innerBoundary], -1, (0, 255, 0), 2)
        
        # Extract and transform playing area
        targetSize = 400
        destPoints = np.array([
            [0, 0],           # top-left
            [targetSize, 0],  # top-right
            [targetSize, targetSize],  # bottom-right
            [0, targetSize]   # bottom-left
        ], dtype=np.float32)
        
        # Sort the corners before transform
        srcPoints = sortCorners(innerBoundary)
        matrix = cv2.getPerspectiveTransform(srcPoints, destPoints)
        playingArea = cv2.warpPerspective(gray, matrix, (targetSize, targetSize))
        
        # Clean up playing area
        cleanedPlayingArea = cv2.dilate(playingArea, kernel, iterations=1)
        cleanedPlayingArea = cv2.morphologyEx(cleanedPlayingArea, cv2.MORPH_CLOSE, kernel)
        cleanedPlayingArea = cv2.morphologyEx(cleanedPlayingArea, cv2.MORPH_OPEN, kernel)
        
        # Detect pieces
        board = detectPiecesInSquares(cleanedPlayingArea)
        
        # Show results
        # cv2.imshow('thresh', thresh)
        # cv2.imshow('morph', morph)
        # cv2.imshow('Chess Board Detection', output)
        cv2.imshow('Starting Image', img)
        cv2.imshow('Grayscale', gray)
        cv2.imshow('Cleaned', cleaned)
        cv2.imshow('Threshold', thresh)
        cv2.imshow('Morph', morph)
        cv2.imshow('Chess Board Detection', output)


        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return board
    
    return None

board = detectChessboardBoundary('images/3.jpg')
