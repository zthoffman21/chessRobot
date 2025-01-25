import requests
import chess

def getBestMove(fen: str) -> str:
    """
    Gets the analysis for a board position from the API.
    
    Args:
        fen: The board position to analyze.
        
    Returns:
        String containing the API response for best move in LAN.
    """
    url = "https://stockfish.online/api/s/v2.php" 
    
    params = {"fen": fen, "depth": 10}
    response = requests.get(url, params=params)

    return response.json()['bestmove'].split()[1]


board = chess.Board()

while True:
    bestMove = getBestMove(board.fen())

    print("Your move:", bestMove)
    board.push(chess.Move.from_uci(bestMove))

    opponentMove = input("Opponent move:")
    board.push(chess.Move.from_uci(opponentMove))