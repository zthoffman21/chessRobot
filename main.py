from visionInput.chessBoardDetection import ChessBoardFinder
from model.model import ChessNet
from model.predict import predictImage
import visionInput.input as input
import stockfish
import torch
import cv2
import math
import chess
from PIL import Image
import time

if __name__ == "__main__":
    prevBoard = [[1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [2,2,2,2,2,2,2,2],
                 [2,2,2,2,2,2,2,2],
                 ]
    board = [[0,0,0,0,0,0,0,0] for x in range(8)]

    boardTracker = chess.Board()

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    model.load_state_dict(torch.load('model/chessClassifier.pth', weights_only=True))
    model.eval()  # Set model to evaluation mode

    # Game loop
    while True:
        # Process image
        finder = ChessBoardFinder()
        frame = cv2.imread("images/15.jpg")
        boardCorners, playCorners, filtered, warpedData = finder.findBoard(frame)

        # Finds the length and height of the found playing area
        if len(playCorners) == 4:
            d1 = math.dist((playCorners[0][0], playCorners[0][1]), (playCorners[1][0], playCorners[1][1]))
            d2 = math.dist((playCorners[1][0], playCorners[1][1]), (playCorners[2][0], playCorners[2][1]))
            print(d1*d2)

        # checks if warpedDate is not none and if the length and width of the playing area are within 10% of each other
        if warpedData and abs(d1 - d2) / ((d1 + d2) / 2) <= 0.1:
            warped, squares = warpedData
            
            for i, ((x1, y1), (x2, y2)) in enumerate(squares):
                squareImg = warped[y1:y2, x1:x2].copy()
                # Convert to PIL Image for consistent preprocessing
                pilImage = Image.fromarray(cv2.cvtColor(squareImg, cv2.COLOR_BGR2RGB))
                
                result, probs = predictImage(model, pilImage, device)
                confidence = max(probs).item() * 100
                print(f'Square {i}: {result} (confidence: {confidence:.1f}%)')
                
                board[i // 8][i % 8] = {'white': 2, 'black': 1, 'empty': 0}[result]
    
        # Player made a move
        if board != prevBoard:
            print("\nBoards changed!")
            for x in board:
                print(x)
            # Finding the move the player made
            moveMade = input.findMovePlayed(previousBoard=prevBoard, currentBoard=board)
            boardTracker.push(chess.Move.from_uci(moveMade))
            print(f'Move made: {moveMade}')

            # Finding the best move to make
            bestMove = stockfish.getBestMove(boardTracker.fen())
            boardTracker.push(chess.Move.from_uci(bestMove))
            print(f'Move made: {bestMove}')

            # ...
            # Code to make move physically
            # ...

            prevBoard = board

        # Wait 5 seconds before checking if player made a move\
        time.sleep(5)