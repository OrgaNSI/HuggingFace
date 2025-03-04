import chess
import chess.engine
import random
import numpy as np
import time
import chess.svg
import cairosvg
from PIL import Image
import io
import math
import pygame
import shutil
import requests

# CONFIGURATION DU CHEMIN DE STOCKFISH (À MODIFIER)
STOCKFISH_PATH = shutil.which("stockfish")
if not STOCKFISH_PATH:
    raise FileNotFoundError("❌ Stockfish non trouvé. Vérifie que Stockfish est installé sur Streamlit Cloud.")

class ChessAI:
    def __init__(self):
        """Initialisation de l'IA et chargement de la Q-table"""
        self.q_table = self.load_q_table()  # Chargement de la Q-table corrigé
        self.alpha = 0.1  # Taux d'apprentissage
        self.gamma = 0.9  # Importance des récompenses futures
        self.epsilon = 1.0  # Exploration initiale
        self.epsilon_decay = 0.999  # Réduction de l'exploration au fil du temps

        print("Lancement de Stockfish...")
        self.stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        time.sleep(1)
        print("Stockfish prêt !")

        # Initialisation de l'affichage avec Pygame
        pygame.init()
        self.window_size = 500
        self.screen = pygame.display.set_mode((self.window_size + 200, self.window_size))
        pygame.display.set_caption("IA d'échecs - Entraînement")

        # Suivi des statistiques de parties
        self.win = 0
        self.loss = 0
        self.draw = 0

    def get_state(self, board):
        """Convertit l’état du plateau en une clé pour la Q-table"""
        return board.fen()

    def choose_move(self, board):
        """Choisit un coup basé sur Q-Learning ou aléatoirement"""
        state = self.get_state(board)
        legal_moves = list(board.legal_moves)

        if state not in self.q_table:
            self.q_table[state] = {move.uci(): 0 for move in legal_moves}
        
        # Stratégie exploration/exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(legal_moves)  # Exploration
        else:
            return max(legal_moves, key=lambda move: self.q_table[state][move.uci()])  # Exploitation

def get_stockfish_move(board):
    """Obtenir un coup depuis une API Stockfish externe"""
    url = "https://lichess.org/api/cloud-eval"  # API Cloud Lichess (basée sur Stockfish)
    fen = board.fen()
    params = {"fen": fen}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        best_move = data.get("pvs", [{}])[0].get("moves", "").split()[0]
        return best_move if best_move else None
    return None

    def get_intermediate_reward(self, board, move):
        """Donne une récompense intermédiaire pour encourager les bons coups"""
        reward = 0
        temp_board = board.copy()  # Crée une copie du plateau pour tester

        if move in temp_board.legal_moves:  # Vérifie que le coup est pseudo-légal
            temp_board.push(move)  # Joue le coup sur la copie

        # 1️⃣ Récompense les captures (selon la valeur de la pièce prise)
        if temp_board.is_capture(move):
            captured_piece = board.piece_at(chess.parse_square(move.uci()[2:4]))
            if captured_piece:
                piece_value = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}.get(captured_piece.symbol().lower(), 0)
                reward += piece_value  # Donne une récompense basée sur la valeur de la pièce

        # 2️⃣ Récompense si l'IA met l'adversaire en échec
        if temp_board.is_check():
            reward += 0.7

        # 3️⃣ Récompense pour le développement des pièces (évite de jouer trop avec les pions)
        if move.uci()[0] in "nbq":
            reward += 0.5  # Encourage l'IA à sortir ses cavaliers et fous

        # 4️⃣ Récompense pour le contrôle du centre
        center_squares = ["d4", "e4", "d5", "e5"]
        if move.uci()[2:4] in center_squares:
            reward += 0.5  # Encourage les coups qui contrôlent le centre

        # 5️⃣ Récompense pour la promotion d'un pion
        if move.promotion:
            reward += 2.0  # Récompense plus élevée pour promouvoir un pion

        # 6️⃣ Pénaliser les mouvements précoces de la Dame (avant le 10e coup)
        if move.uci()[0] == "q" and board.fullmove_number < 10:
            reward -= 1.0  # Pénalisation si la Dame bouge trop tôt

        # 7️⃣ Récompense plus forte pour sortir Cavaliers et Fous en début de partie
        if move.uci()[0] in "nb" and board.fullmove_number < 10:
            reward += 0.7

        return reward


    def update_q_table(self, state, move, reward, next_state):
        """Met à jour la table Q en ajoutant les nouveaux coups au besoin"""
        move_uci = move.uci()  # Définir `move_uci` AVANT de l'utiliser

        # Vérifier si l'état courant existe dans la Q-table, sinon l'initialiser
        if state not in self.q_table:
            self.q_table[state] = {}

        # Vérifier si le coup n'est pas encore dans la Q-table et l'ajouter
        if move_uci not in self.q_table[state]:
            self.q_table[state][move_uci] = 0  # Initialise la valeur du coup

        # Vérifier si l'état suivant est connu, sinon l'initialiser avec des valeurs neutres
        if next_state not in self.q_table:
            self.q_table[next_state] = {}

        # Trouver la meilleure valeur future (max de la Q-table)
        max_future_q = max(self.q_table[next_state].values(), default=0) if next_state in self.q_table else 0

        # Sauvegarde de l'ancienne valeur pour voir l'évolution
        old_value = self.q_table[state][move_uci]

        # Mise à jour de la Q-table avec la récompense et la meilleure estimation future
        self.q_table[state][move_uci] += self.alpha * (reward + self.gamma * max_future_q - old_value)

        # Affichage pour vérifier la mise à jour
        print(f"🔄 Q-table update: {state} -> {move_uci} | Old: {old_value:.4f} | New: {self.q_table[state][move_uci]:.4f}")


    def get_reward(self, board):
        """Attribue une récompense basée sur la situation du jeu"""
        if board.is_checkmate():
            return 1 if board.turn == chess.BLACK else -1  # Victoire = +1, Défaite = -1
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.5  # Égalité = 0.5
        return 0  # Autres situations

    def handle_pygame_events(self):
        """Gère les événements Pygame pour éviter les crashs si la fenêtre se ferme"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Fermeture demandée. Sauvegarde de la Q-table...")
                self.save_q_table()
                pygame.quit()
                exit()  # Quitter proprement le programme

    def display_board(self, board):
        """Affiche le plateau d'échecs avec Pygame et gère les événements"""
        self.handle_pygame_events()  # Gérer les événements avant de dessiner

        svg_data = chess.svg.board(board=board).encode("utf-8")
        png_data = cairosvg.svg2png(bytestring=svg_data)  # Convertir SVG en PNG
        image = Image.open(io.BytesIO(png_data))
        image = image.resize((self.window_size, self.window_size))  # Adapter la taille

        pygame_surface = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
        self.screen.blit(pygame_surface, (0, 0))

        # Calcul du taux de progression basé sur le ratio de victoires
        total_games = self.win + self.loss + self.draw
        progression = int((self.win / total_games) * 100) if total_games > 0 else 0

        # Affichage de la progression
        font = pygame.font.Font(None, 24)
        text = font.render(f"Progression : {progression}%", True, (255, 255, 255))
        self.screen.fill((0, 0, 0), (self.window_size, 0, 200, self.window_size))  # Fond noir à droite
        self.screen.blit(text, (self.window_size + 20, 20))

        pygame.display.flip()
        time.sleep(0.5)  # Pause pour voir les coups


    def estimate_elo(self, base_elo=1000):
        """Estime le Elo de l'IA en fonction des victoires/défaites contre Stockfish"""
        W = max(self.win, 1)  # Évite la division par 0 en mettant au moins 1 victoire
        L = max(self.loss, 1)  # Évite la division par 0

        estimated_elo = base_elo + 400 * math.log10(W / L)
        return int(estimated_elo)


    def train(self, num_games=10):
        """Entraîne l'IA en jouant contre Stockfish avec affichage dynamique"""
        print(f"Début de l'entraînement sur {num_games} parties...")

        for game in range(num_games):
            board = chess.Board()
            state = self.get_state(board)

            print(f"\n🔹 Partie {game + 1}/{num_games}")
            self.display_board(board)  # Affichage initial

            while not board.is_game_over():
                self.handle_pygame_events()  # Vérifier si la fenêtre est fermée

                if board.fullmove_number == 1 and random.random() < 0.5:  
                    board.push(random.choice(list(board.legal_moves)))  # Force un coup d'ouverture aléatoire

                # Coup de l'IA
                move = self.choose_move(board)
                board.push(move)
                print(f"🤖 IA joue : {move.uci()} ({move})")

                # Calcul de la récompense intermédiaire
                intermediate_reward = self.get_intermediate_reward(board, move)
                reward = self.get_reward(board) + intermediate_reward
                print(f"💰 Récompense intermédiaire pour {move.uci()} : {intermediate_reward:.2f}")

                next_state = self.get_state(board)
                self.update_q_table(state, move, reward, next_state)


                if board.is_game_over():
                    break  # Arrêter la partie si elle est finie

                # Coup de Stockfish
                stockfish_move_uci = get_stockfish_move(board)  # Utilise l'API Lichess
                if stockfish_move_uci:
                    stockfish_move = chess.Move.from_uci(stockfish_move_uci)
                    if stockfish_move in board.legal_moves:  # Vérifie que le coup est valide
                        board.push(stockfish_move)
                        print(f"♟️ Stockfish joue : {stockfish_move_uci} ({stockfish_move})")
                    else:
                        print("❌ Coup illégal détecté, Stockfish n'a pas joué.")
                else:
                    print("⚠️ Impossible d'obtenir un coup de Stockfish.")

                print(f"♟️ Stockfish joue : {stockfish_move.uci()} ({stockfish_move})")
                self.display_board(board)  # Affichage après le coup de Stockfish

                state = self.get_state(board)

            # Mise à jour des statistiques
            result = board.result()
            if result == "1-0":
                self.win += 1
            elif result == "0-1":
                self.loss += 1
            else:
                self.draw += 1

            # Sauvegarde toutes les 10 parties
            if (game + 1) % 10 == 0:
                self.save_q_table()

            # Réduction progressive de l'exploration
            self.epsilon *= self.epsilon_decay

            # Estimation du Elo de l'IA après l'entraînement
            elo = self.estimate_elo()
            print(f"\n🔹 Estimation du Elo de l'IA : {elo} Elo")


        print(f"\n🔹 Résultats après {num_games} parties :")
        print(f"🏆 Victoires : {self.win}")
        print(f"💀 Défaites : {self.loss}")
        print(f"🤝 Nulles : {self.draw}")

        self.stockfish.quit()
        self.save_q_table()
        pygame.quit()

    def save_q_table(self, filename="q_table.npy"):
        """Sauvegarde la Q-table toutes les 10 parties"""
        np.save(filename, self.q_table)
        print("Q-table sauvegardée.")

    def load_q_table(self, filename="q_table.npy"):
        """Charge une table Q existante"""
        try:
            return np.load(filename, allow_pickle=True).item()
        except FileNotFoundError:
            print("Aucune Q-table trouvée, démarrage à zéro.")
            return {}

# Lancer l'entraînement
if __name__ == "__main__":
    ai = ChessAI()
    ai.train(10)  # Nb parties à entraîner
