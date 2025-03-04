import streamlit as st
import chess
import chess.svg
import chess.engine

# Charger Stockfish (hébergé localement ou sur un serveur distant)
STOCKFISH_PATH = "stockfish"  # Sur Streamlit, il faut l'ajouter manuellement
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

st.title("IA d'échecs en ligne")

board = chess.Board()

if st.button("Faire jouer l'IA"):
    result = engine.play(board, chess.engine.Limit(time=0.5))
    board.push(result.move)

# Afficher l'échiquier en SVG
st.image(chess.svg.board(board=board), use_column_width=True)
