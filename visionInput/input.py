def findMovePlayed(previousBoard: list[list[int]], currentBoard: list[list[int]]) -> str:
    """
    Compares previous and current board positions to determine which move was played

    Returns:
        str: Long Algebraic Notation of the move made
    """
    key = "abcdefgh"

    # Check for castling
    if previousBoard[0][0] != currentBoard[0][0] and previousBoard[0][2] != currentBoard[0][2] and previousBoard[0][3] != currentBoard[0][3] and previousBoard[0][4] != currentBoard[0][4]:
        return "e8c8"
    elif previousBoard[0][4] != currentBoard[0][4] and previousBoard[0][5] != currentBoard[0][5] and previousBoard[0][6] != currentBoard[0][6] and previousBoard[0][7] != currentBoard[0][7]:
        return "e8g8"
    elif previousBoard[7][0] != currentBoard[7][0] and previousBoard[7][2] != currentBoard[7][2] and previousBoard[7][3] != currentBoard[7][3] and previousBoard[7][4] != currentBoard[7][4]:
        return "e1c1"
    elif previousBoard[7][4] != currentBoard[7][4] and previousBoard[7][5] != currentBoard[7][5] and previousBoard[7][6] != currentBoard[7][6] and previousBoard[7][7] != currentBoard[7][7]:
        return "e1g1"

    endPosition = ""
    startPosition = ""

    for row in range(8):
        for column in range(8):
            if previousBoard[row][column] != currentBoard[row][column]:
                if currentBoard[row][column] == 0: 
                    startPosition = key[column] + str(8 - row) # 8 - row is used to reverse from counting down the rows (like a 2d array) to counting up (like in chess)
                else:
                    endPosition = key[column] + str(8 - row)

            if endPosition != "" and startPosition != "":
                return startPosition + endPosition
    
    return startPosition + endPosition

def rotateBoard(board: list[list[int]]) -> list[list[int]]:
    """
    Rotates the given board by 180. Needed because the correct orientation is needed so the findMovePlayed function can assign the spaces correctly.

    Args:
        board (list[list[int]]): A 2d array showing the current board with 0 = empty square, 1 = black piece, and 2 = white piece. This board is the original.

    Returns:
        list[list[int]]: A 2d array showing the current board with 0 = empty square, 1 = black piece, and 2 = white piece. This board is flipped 180.
    """
    tempBoard = [[] for x in range(len(board))]

    for row in range(7, -1, -1):
        tempBoard[7-row] = board[row].copy()
        tempBoard[7-row].reverse()

    return tempBoard

if __name__ == "__main__":
    previousBoard = [
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [2,2,2,2,2,2,2,2],
        [2,2,2,2,2,2,2,2],
    ]

    currentBoard = [
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,2,0,0,0],
        [0,0,0,0,0,0,0,0],
        [2,2,2,2,0,2,2,2],
        [2,0,2,2,2,2,2,0],
    ]