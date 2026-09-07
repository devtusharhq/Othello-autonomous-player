import tkinter as tk
from tkinter import messagebox
from copy import deepcopy
import math
import time


# ============================================================
# OTHELLO / REVERSI AUTONOMOUS PLAYER
# AI: MINIMAX + ALPHA-BETA PRUNING
# GUI: TKINTER
# ============================================================


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

BOARD_SIZE = 8

EMPTY = 0
BLACK = 1
WHITE = -1

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

POSITION_WEIGHTS = [
    [120, -25,  20,   5,   5,  20, -25, 120],
    [-25, -45,  -5,  -5,  -5,  -5, -45, -25],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-25, -45,  -5,  -5,  -5,  -5, -45, -25],
    [120, -25,  20,   5,   5,  20, -25, 120]
]

CORNERS = [
    (0, 0),
    (0, 7),
    (7, 0),
    (7, 7)
]


# ============================================================
# GAME ENGINE
# ============================================================

class OthelloGame:

    def __init__(self):
        self.reset()

    def reset(self):

        self.board = [
            [EMPTY for _ in range(BOARD_SIZE)]
            for _ in range(BOARD_SIZE)
        ]

        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE

        self.current_player = BLACK

    def inside(self, row, col):
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def is_valid_move(self, row, col, player):

        if not self.inside(row, col):
            return False

        if self.board[row][col] != EMPTY:
            return False

        opponent = -player

        for dr, dc in DIRECTIONS:

            r = row + dr
            c = col + dc

            found_opponent = False

            while self.inside(r, c) and self.board[r][c] == opponent:
                found_opponent = True
                r += dr
                c += dc

            if (
                found_opponent
                and self.inside(r, c)
                and self.board[r][c] == player
            ):
                return True

        return False

    def get_valid_moves(self, player):

        moves = []

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):

                if self.is_valid_move(row, col, player):
                    moves.append((row, col))

        return moves

    def make_move(self, row, col, player):

        if not self.is_valid_move(row, col, player):
            return False

        opponent = -player

        self.board[row][col] = player

        for dr, dc in DIRECTIONS:

            captured = []

            r = row + dr
            c = col + dc

            while (
                self.inside(r, c)
                and self.board[r][c] == opponent
            ):
                captured.append((r, c))
                r += dr
                c += dc

            if (
                captured
                and self.inside(r, c)
                and self.board[r][c] == player
            ):
                for rr, cc in captured:
                    self.board[rr][cc] = player

        return True

    def count_discs(self, player):

        return sum(
            row.count(player)
            for row in self.board
        )

    def empty_count(self):

        return sum(
            row.count(EMPTY)
            for row in self.board
        )

    def is_game_over(self):

        if self.empty_count() == 0:
            return True

        if not self.get_valid_moves(BLACK) and not self.get_valid_moves(WHITE):
            return True

        return False

    def winner(self):

        black = self.count_discs(BLACK)
        white = self.count_discs(WHITE)

        if black > white:
            return BLACK

        if white > black:
            return WHITE

        return 0


# ============================================================
# AI ENGINE
# ============================================================

class OthelloAI:

    def __init__(self, depth=4):

        self.depth = depth
        self.nodes_searched = 0
        self.pruned_branches = 0
        self.last_thinking_time = 0

    def copy_board(self, board):
        return [row[:] for row in board]

    def valid_moves(self, board, player):

        moves = []

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):

                if board[row][col] != EMPTY:
                    continue

                if self.is_valid_move(board, row, col, player):
                    moves.append((row, col))

        return moves

    def is_valid_move(self, board, row, col, player):

        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False

        if board[row][col] != EMPTY:
            return False

        opponent = -player

        for dr, dc in DIRECTIONS:

            r = row + dr
            c = col + dc

            found_opponent = False

            while (
                0 <= r < BOARD_SIZE
                and 0 <= c < BOARD_SIZE
                and board[r][c] == opponent
            ):
                found_opponent = True
                r += dr
                c += dc

            if (
                found_opponent
                and 0 <= r < BOARD_SIZE
                and 0 <= c < BOARD_SIZE
                and board[r][c] == player
            ):
                return True

        return False

    def apply_move(self, board, move, player):

        new_board = self.copy_board(board)

        row, col = move
        opponent = -player

        new_board[row][col] = player

        for dr, dc in DIRECTIONS:

            captured = []

            r = row + dr
            c = col + dc

            while (
                0 <= r < BOARD_SIZE
                and 0 <= c < BOARD_SIZE
                and new_board[r][c] == opponent
            ):
                captured.append((r, c))
                r += dr
                c += dc

            if (
                captured
                and 0 <= r < BOARD_SIZE
                and 0 <= c < BOARD_SIZE
                and new_board[r][c] == player
            ):

                for rr, cc in captured:
                    new_board[rr][cc] = player

        return new_board

    # ========================================================
    # HEURISTIC EVALUATION
    # ========================================================

    def evaluate(self, board, player):

        opponent = -player

        player_count = 0
        opponent_count = 0
        positional_score = 0

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):

                cell = board[row][col]

                if cell == player:

                    player_count += 1
                    positional_score += POSITION_WEIGHTS[row][col]

                elif cell == opponent:

                    opponent_count += 1
                    positional_score -= POSITION_WEIGHTS[row][col]

        # Corner control
        player_corners = 0
        opponent_corners = 0

        for row, col in CORNERS:

            if board[row][col] == player:
                player_corners += 1

            elif board[row][col] == opponent:
                opponent_corners += 1

        corner_score = 25 * (
            player_corners - opponent_corners
        )

        # Mobility
        player_moves = len(
            self.valid_moves(board, player)
        )

        opponent_moves = len(
            self.valid_moves(board, opponent)
        )

        if player_moves + opponent_moves != 0:

            mobility_score = 100 * (
                player_moves - opponent_moves
            ) / (
                player_moves + opponent_moves
            )

        else:
            mobility_score = 0

        # Disc difference
        total_discs = player_count + opponent_count

        if total_discs != 0:

            disc_score = 100 * (
                player_count - opponent_count
            ) / total_discs

        else:
            disc_score = 0

        score = (
            positional_score
            + corner_score * 3
            + mobility_score * 2
            + disc_score
        )

        return score

    # ========================================================
    # MINIMAX + ALPHA-BETA PRUNING
    # ========================================================

    def minimax(
        self,
        board,
        depth,
        player,
        maximizing_player,
        alpha,
        beta
    ):

        self.nodes_searched += 1

        current_moves = self.valid_moves(
            board,
            player
        )

        opponent_moves = self.valid_moves(
            board,
            -player
        )

        if depth == 0 or (
            not current_moves
            and not opponent_moves
        ):

            return self.evaluate(
                board,
                maximizing_player
            ), None

        if not current_moves:

            score, _ = self.minimax(
                board,
                depth - 1,
                -player,
                maximizing_player,
                alpha,
                beta
            )

            return score, None

        maximizing = player == maximizing_player

        best_move = None

        if maximizing:

            best_score = -math.inf

            for move in self.order_moves(
                board,
                current_moves,
                player
            ):

                child = self.apply_move(
                    board,
                    move,
                    player
                )

                score, _ = self.minimax(
                    child,
                    depth - 1,
                    -player,
                    maximizing_player,
                    alpha,
                    beta
                )

                if score > best_score:

                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)

                if beta <= alpha:

                    self.pruned_branches += 1
                    break

            return best_score, best_move

        else:

            best_score = math.inf

            for move in self.order_moves(
                board,
                current_moves,
                player
            ):

                child = self.apply_move(
                    board,
                    move,
                    player
                )

                score, _ = self.minimax(
                    child,
                    depth - 1,
                    -player,
                    maximizing_player,
                    alpha,
                    beta
                )

                if score < best_score:

                    best_score = score
                    best_move = move

                beta = min(beta, best_score)

                if beta <= alpha:

                    self.pruned_branches += 1
                    break

            return best_score, best_move

    def order_moves(self, board, moves, player):

        def move_value(move):

            row, col = move

            value = POSITION_WEIGHTS[row][col]

            if move in CORNERS:
                value += 1000

            return value

        return sorted(
            moves,
            key=move_value,
            reverse=True
        )

    def get_best_move(self, board, player):

        start_time = time.time()

        self.nodes_searched = 0
        self.pruned_branches = 0

        moves = self.valid_moves(
            board,
            player
        )

        if not moves:

            self.last_thinking_time = (
                time.time() - start_time
            )

            return None

        # Easy mode deliberately makes weaker strategic choices.
        # This keeps the mode very beginner-friendly while the
        # stronger modes continue using the normal minimax strategy.
        if self.depth == 1:

            def easy_move_value(move):
                row, col = move
                return POSITION_WEIGHTS[row][col]

            best_move = min(
                moves,
                key=easy_move_value
            )

            self.last_thinking_time = (
                time.time() - start_time
            )

            return best_move

        # Immediate corner priority for Normal, Hard and Expert
        for move in moves:

            if move in CORNERS:

                self.last_thinking_time = (
                    time.time() - start_time
                )

                return move

        _, best_move = self.minimax(
            board,
            self.depth,
            player,
            player,
            -math.inf,
            math.inf
        )

        self.last_thinking_time = (
            time.time() - start_time
        )

        return best_move


# ============================================================
# TKINTER GUI
# ============================================================

class OthelloGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "OTHELLO // AUTONOMOUS PLAYER"
        )

        # Increased default height slightly and improved minimum size
        self.root.geometry("1120x780")
        self.root.minsize(950, 680)

        # ----------------------------------------------------
        # THEME
        # ----------------------------------------------------

        self.bg = "#0b1220"
        self.panel = "#111b2e"
        self.panel2 = "#17243a"

        self.board_bg = "#064e3b"
        self.grid_color = "#0f766e"

        self.text = "#f8fafc"
        self.muted = "#94a3b8"

        self.black_color = "#111827"
        self.white_color = "#f8fafc"

        self.accent = "#38bdf8"
        self.green = "#22c55e"
        self.orange = "#f59e0b"
        self.red = "#ef4444"

        self.root.configure(
            bg=self.bg
        )

        # ----------------------------------------------------
        # GAME STATE
        # ----------------------------------------------------

        self.game = OthelloGame()

        self.ai_player = WHITE

        self.ai = OthelloAI(depth=4)

        self.history = []

        self.move_number = 0

        self.last_move = None

        self.ai_thinking = False

        self.show_legal_moves = True

        # Prevent game-over dialog from being shown more than once
        self.game_over_shown = False

        # ----------------------------------------------------
        # BUILD UI
        # ----------------------------------------------------

        self.build_header()
        self.build_main_area()
        self.build_status_bar()

        self.draw_board()
        self.update_information()

    # ========================================================
    # HEADER
    # ========================================================

    def build_header(self):

        header = tk.Frame(
            self.root,
            bg=self.bg
        )

        header.pack(
            fill="x",
            padx=28,
            pady=(14, 6)
        )

        title = tk.Label(
            header,
            text="OTHELLO",
            font=("Segoe UI", 26, "bold"),
            fg=self.text,
            bg=self.bg
        )

        title.pack(side="left")

        subtitle = tk.Label(
            header,
            text="  //  AUTONOMOUS PLAYER",
            font=("Consolas", 12, "bold"),
            fg=self.accent,
            bg=self.bg
        )

        subtitle.pack(
            side="left",
            pady=(8, 0)
        )

        self.turn_label = tk.Label(
            header,
            text="",
            font=("Segoe UI", 11, "bold"),
            fg=self.green,
            bg=self.bg
        )

        self.turn_label.pack(
            side="right",
            pady=8
        )

    # ========================================================
    # MAIN AREA
    # ========================================================

    def build_main_area(self):

        main = tk.Frame(
            self.root,
            bg=self.bg
        )

        main.pack(
            fill="both",
            expand=True,
            padx=28,
            pady=6
        )

        # ----------------------------------------------------
        # LEFT BOARD
        # ----------------------------------------------------

        board_container = tk.Frame(
            main,
            bg=self.panel,
            highlightbackground="#233653",
            highlightthickness=1
        )

        board_container.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 14)
        )

        board_title = tk.Label(
            board_container,
            text="GAME BOARD",
            font=("Segoe UI", 10, "bold"),
            fg=self.muted,
            bg=self.panel
        )

        board_title.pack(
            anchor="w",
            padx=18,
            pady=(10, 2)
        )

        self.canvas = tk.Canvas(
            board_container,
            bg=self.board_bg,
            highlightthickness=0
        )

        self.canvas.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(8, 18)
        )

        self.canvas.bind(
            "<Button-1>",
            self.board_click
        )

        self.canvas.bind(
            "<Configure>",
            lambda event: self.draw_board()
        )

        # ----------------------------------------------------
        # RIGHT PANEL
        # ----------------------------------------------------

        side = tk.Frame(
            main,
            bg=self.panel,
            width=310
        )

        side.pack(
            side="right",
            fill="both"
        )

        side.pack_propagate(False)

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score_title = tk.Label(
            side,
            text="SCORE",
            font=("Segoe UI", 9, "bold"),
            fg=self.muted,
            bg=self.panel
        )

        score_title.pack(
            anchor="w",
            padx=20,
            pady=(14, 5)
        )

        score_frame = tk.Frame(
            side,
            bg=self.panel2
        )

        score_frame.pack(
            fill="x",
            padx=20
        )

        self.black_score = tk.Label(
            score_frame,
            text="BLACK\n2",
            font=("Segoe UI", 16, "bold"),
            fg=self.text,
            bg=self.panel2,
            justify="center"
        )

        self.black_score.pack(
            side="left",
            expand=True,
            pady=10
        )

        self.white_score = tk.Label(
            score_frame,
            text="WHITE\n2",
            font=("Segoe UI", 16, "bold"),
            fg=self.text,
            bg=self.panel2,
            justify="center"
        )

        self.white_score.pack(
            side="right",
            expand=True,
            pady=10
        )

        # ----------------------------------------------------
        # AUTONOMOUS AI
        # ----------------------------------------------------

        ai_title = tk.Label(
            side,
            text="AUTONOMOUS AI",
            font=("Segoe UI", 9, "bold"),
            fg=self.muted,
            bg=self.panel
        )

        ai_title.pack(
            anchor="w",
            padx=20,
            pady=(12, 3)
        )

        self.ai_status = tk.Label(
            side,
            text="READY",
            font=("Consolas", 13, "bold"),
            fg=self.green,
            bg=self.panel
        )

        self.ai_status.pack(
            anchor="w",
            padx=20
        )

        # ----------------------------------------------------
        # AI DEPTH
        # ----------------------------------------------------

        difficulty_frame = tk.Frame(
            side,
            bg=self.panel
        )

        difficulty_frame.pack(
            fill="x",
            padx=20,
            pady=(9, 2)
        )

        tk.Label(
            difficulty_frame,
            text="AI DEPTH",
            font=("Segoe UI", 8, "bold"),
            fg=self.muted,
            bg=self.panel
        ).pack(
            anchor="w",
            pady=(0, 2)
        )

        self.depth_var = tk.IntVar(
            value=4
        )

        depth_options = [
            ("Easy", 1),
            ("Normal", 3),
            ("Hard", 5),
            ("Expert", 6)
        ]

        # Put difficulty levels horizontally in two columns
        # to save vertical space and improve alignment.
        difficulty_options = tk.Frame(
            difficulty_frame,
            bg=self.panel
        )

        difficulty_options.pack(
            fill="x"
        )

        for index, (text, depth) in enumerate(depth_options):

            rb = tk.Radiobutton(
                difficulty_options,
                text=text,
                variable=self.depth_var,
                value=depth,
                command=self.change_depth,
                bg=self.panel,
                fg=self.text,
                activebackground=self.panel,
                activeforeground=self.accent,
                selectcolor=self.panel2,
                font=("Segoe UI", 8),
                anchor="w"
            )

            rb.grid(
                row=index // 2,
                column=index % 2,
                sticky="w",
                padx=(0, 20),
                pady=1
            )

        difficulty_options.columnconfigure(
            0,
            weight=1
        )

        difficulty_options.columnconfigure(
            1,
            weight=1
        )

        # ----------------------------------------------------
        # AI ANALYTICS
        # ----------------------------------------------------

        stats_title = tk.Label(
            side,
            text="AI ANALYTICS",
            font=("Segoe UI", 9, "bold"),
            fg=self.muted,
            bg=self.panel
        )

        stats_title.pack(
            anchor="w",
            padx=20,
            pady=(10, 4)
        )

        self.stats_label = tk.Label(
            side,
            text=(
                "Nodes searched: 0\n"
                "Branches pruned: 0\n"
                "Thinking time: 0.000s"
            ),
            font=("Consolas", 8),
            fg=self.text,
            bg=self.panel2,
            justify="left",
            anchor="w",
            padx=12,
            pady=8
        )

        self.stats_label.pack(
            fill="x",
            padx=20
        )

        # ----------------------------------------------------
        # CONTROLS
        # ----------------------------------------------------

        controls_title = tk.Label(
            side,
            text="CONTROLS",
            font=("Segoe UI", 9, "bold"),
            fg=self.muted,
            bg=self.panel
        )

        controls_title.pack(
            anchor="w",
            padx=20,
            pady=(10, 4)
        )

        controls_frame = tk.Frame(
            side,
            bg=self.panel
        )

        controls_frame.pack(
            fill="x",
            padx=20,
            pady=(0, 12)
        )

        self.create_button(
            controls_frame,
            "NEW GAME",
            self.new_game
        )

        self.create_button(
            controls_frame,
            "UNDO MOVE",
            self.undo_move
        )

        self.create_button(
            controls_frame,
            "TOGGLE LEGAL MOVES",
            self.toggle_legal_moves
        )

    # ========================================================
    # BUTTON
    # ========================================================

    def create_button(
        self,
        parent,
        text,
        command
    ):

        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 8, "bold"),
            bg=self.panel2,
            fg=self.text,
            activebackground="#243754",
            activeforeground=self.accent,
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=6,
            bd=0
        )

        button.pack(
            fill="x",
            pady=2
        )

    # ========================================================
    # STATUS BAR
    # ========================================================

    def build_status_bar(self):

        self.status = tk.Label(
            self.root,
            text="Welcome to Othello.",
            font=("Consolas", 9),
            fg=self.muted,
            bg=self.bg,
            anchor="w"
        )

        self.status.pack(
            fill="x",
            padx=28,
            pady=(3, 10)
        )

    # ========================================================
    # DRAW BOARD
    # ========================================================

    def draw_board(self):

        if not hasattr(self, "canvas"):
            return

        self.canvas.delete("all")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width < 100 or height < 100:
            return

        size = min(
            width,
            height
        )

        x_offset = (width - size) / 2
        y_offset = (height - size) / 2

        cell = size / BOARD_SIZE

        self.canvas.create_rectangle(
            x_offset,
            y_offset,
            x_offset + size,
            y_offset + size,
            fill=self.board_bg,
            outline=""
        )

        # Grid
        for i in range(BOARD_SIZE + 1):

            x = x_offset + i * cell
            y = y_offset + i * cell

            self.canvas.create_line(
                x,
                y_offset,
                x,
                y_offset + size,
                fill=self.grid_color,
                width=1
            )

            self.canvas.create_line(
                x_offset,
                y,
                x_offset + size,
                y,
                fill=self.grid_color,
                width=1
            )

        # Coordinates
        for i in range(BOARD_SIZE):

            self.canvas.create_text(
                x_offset + i * cell + cell / 2,
                y_offset + 10,
                text=str(i + 1),
                fill="#a7f3d0",
                font=("Consolas", 8)
            )

            self.canvas.create_text(
                x_offset + 10,
                y_offset + i * cell + cell / 2,
                text=chr(65 + i),
                fill="#a7f3d0",
                font=("Consolas", 8)
            )

        # Legal moves
        if self.show_legal_moves and not self.ai_thinking:

            legal_moves = self.game.get_valid_moves(
                self.game.current_player
            )

            for row, col in legal_moves:

                cx = (
                    x_offset
                    + col * cell
                    + cell / 2
                )

                cy = (
                    y_offset
                    + row * cell
                    + cell / 2
                )

                radius = max(
                    3,
                    cell * 0.08
                )

                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill="#86efac",
                    outline=""
                )

        # Pieces
        for row in range(BOARD_SIZE):

            for col in range(BOARD_SIZE):

                value = self.game.board[row][col]

                if value == EMPTY:
                    continue

                cx = (
                    x_offset
                    + col * cell
                    + cell / 2
                )

                cy = (
                    y_offset
                    + row * cell
                    + cell / 2
                )

                radius = cell * 0.38

                # Last move glow
                if self.last_move == (row, col):

                    self.canvas.create_oval(
                        cx - radius - 5,
                        cy - radius - 5,
                        cx + radius + 5,
                        cy + radius + 5,
                        outline=self.accent,
                        width=3
                    )

                if value == BLACK:

                    self.canvas.create_oval(
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                        fill=self.black_color,
                        outline="#475569",
                        width=2
                    )

                    self.canvas.create_oval(
                        cx - radius * .65,
                        cy - radius * .65,
                        cx - radius * .35,
                        cy - radius * .35,
                        fill="#64748b",
                        outline=""
                    )

                else:

                    self.canvas.create_oval(
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                        fill=self.white_color,
                        outline="#cbd5e1",
                        width=2
                    )

                    self.canvas.create_oval(
                        cx - radius * .65,
                        cy - radius * .65,
                        cx - radius * .35,
                        cy - radius * .35,
                        fill="#ffffff",
                        outline=""
                    )

        self.root.after(
            20,
            self.update_information
        )

    # ========================================================
    # BOARD CLICK
    # ========================================================

    def board_click(self, event):

        if self.ai_thinking:
            return

        if self.game.is_game_over():
            return

        if self.game.current_player != BLACK:
            return

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        size = min(
            width,
            height
        )

        x_offset = (width - size) / 2
        y_offset = (height - size) / 2

        cell = size / BOARD_SIZE

        col = int(
            (event.x - x_offset) / cell
        )

        row = int(
            (event.y - y_offset) / cell
        )

        if not (
            0 <= row < BOARD_SIZE
            and 0 <= col < BOARD_SIZE
        ):
            return

        if not self.game.is_valid_move(
            row,
            col,
            BLACK
        ):

            self.status.config(
                text="Invalid move. Select a highlighted position."
            )

            return

        self.save_state()

        self.game.make_move(
            row,
            col,
            BLACK
        )

        self.last_move = (row, col)

        self.move_number += 1

        self.game.current_player = WHITE

        self.draw_board()
        self.update_information()

        self.check_turn()

    # ========================================================
    # TURN MANAGEMENT
    # ========================================================

    def check_turn(self):

        if self.game.is_game_over():

            self.end_game()

            return

        current = self.game.current_player

        moves = self.game.get_valid_moves(
            current
        )

        if not moves:

            player_name = (
                "Black"
                if current == BLACK
                else "White"
            )

            self.status.config(
                text=f"{player_name} has no legal moves. Turn passed."
            )

            self.game.current_player = -current

            self.root.after(
                900,
                self.check_turn
            )

            return

        if current == WHITE:

            self.start_ai_turn()

    # ========================================================
    # AI TURN
    # ========================================================

    def start_ai_turn(self):

        self.ai_thinking = True

        self.ai_status.config(
            text="THINKING...",
            fg=self.orange
        )

        self.turn_label.config(
            text="AI IS ANALYZING",
            fg=self.orange
        )

        self.status.config(
            text="Autonomous AI is evaluating possible moves..."
        )

        self.draw_board()

        self.root.after(
            100,
            self.perform_ai_move
        )

    def perform_ai_move(self):

        board_copy = deepcopy(
            self.game.board
        )

        best_move = self.ai.get_best_move(
            board_copy,
            WHITE
        )

        if best_move is None:

            self.ai_thinking = False

            self.game.current_player = BLACK

            self.check_turn()

            return

        self.save_state()

        row, col = best_move

        self.game.make_move(
            row,
            col,
            WHITE
        )

        self.last_move = best_move

        self.move_number += 1

        self.game.current_player = BLACK

        self.ai_thinking = False

        self.ai_status.config(
            text="MOVE COMPLETE",
            fg=self.green
        )

        self.status.config(
            text=(
                f"AI selected {chr(65 + col)}{row + 1} "
                f"after evaluating "
                f"{self.ai.nodes_searched:,} positions."
            )
        )

        self.update_ai_statistics()

        self.draw_board()
        self.update_information()

        self.root.after(
            500,
            self.check_turn
        )

    # ========================================================
    # SAVE STATE
    # ========================================================

    def save_state(self):

        self.history.append(
            (
                deepcopy(self.game.board),
                self.game.current_player,
                self.last_move,
                self.move_number
            )
        )

        if len(self.history) > 30:

            self.history.pop(0)

    # ========================================================
    # UNDO
    # ========================================================

    def undo_move(self):

        if self.ai_thinking:
            return

        if not self.history:

            self.status.config(
                text="There are no moves to undo."
            )

            return

        if len(self.history) >= 2:

            self.history.pop()
            state = self.history.pop()

        else:

            state = self.history.pop()

        (
            board,
            current_player,
            last_move,
            move_number
        ) = state

        self.game.board = deepcopy(
            board
        )

        self.game.current_player = (
            current_player
        )

        self.last_move = last_move

        self.move_number = move_number

        self.game_over_shown = False

        self.ai_status.config(
            text="READY",
            fg=self.green
        )

        self.status.config(
            text="Move undone."
        )

        self.draw_board()
        self.update_information()

    # ========================================================
    # NEW GAME
    # ========================================================

    def new_game(self):

        if self.ai_thinking:
            return

        self.game.reset()

        self.history.clear()

        self.move_number = 0

        self.last_move = None

        self.game_over_shown = False

        self.ai.nodes_searched = 0
        self.ai.pruned_branches = 0
        self.ai.last_thinking_time = 0

        self.ai_status.config(
            text="READY",
            fg=self.green
        )

        self.stats_label.config(
            text=(
                "Nodes searched: 0\n"
                "Branches pruned: 0\n"
                "Thinking time: 0.000s"
            )
        )

        self.status.config(
            text="New game started. Your turn."
        )

        self.draw_board()
        self.update_information()

    # ========================================================
    # CHANGE DIFFICULTY
    # ========================================================

    def change_depth(self):

        self.ai.depth = self.depth_var.get()

        depth = self.ai.depth

        difficulty = {
            1: "Easy",
            3: "Normal",
            5: "Hard",
            6: "Expert"
        }.get(depth, "Custom")

        self.status.config(
            text=(
                f"AI difficulty changed to "
                f"{difficulty} (depth {depth})."
            )
        )

    # ========================================================
    # TOGGLE LEGAL MOVES
    # ========================================================

    def toggle_legal_moves(self):

        self.show_legal_moves = (
            not self.show_legal_moves
        )

        state = (
            "shown"
            if self.show_legal_moves
            else "hidden"
        )

        self.status.config(
            text=f"Legal moves are now {state}."
        )

        self.draw_board()

    # ========================================================
    # UPDATE INFORMATION
    # ========================================================

    def update_information(self):

        black = self.game.count_discs(
            BLACK
        )

        white = self.game.count_discs(
            WHITE
        )

        self.black_score.config(
            text=f"BLACK\n{black}"
        )

        self.white_score.config(
            text=f"WHITE\n{white}"
        )

        if self.game.is_game_over():

            self.turn_label.config(
                text="GAME OVER",
                fg=self.red
            )

        elif self.ai_thinking:

            self.turn_label.config(
                text="AI THINKING",
                fg=self.orange
            )

        elif self.game.current_player == BLACK:

            self.turn_label.config(
                text="YOUR TURN • BLACK",
                fg=self.green
            )

        else:

            self.turn_label.config(
                text="AI TURN • WHITE",
                fg=self.accent
            )

    # ========================================================
    # AI STATISTICS
    # ========================================================

    def update_ai_statistics(self):

        self.stats_label.config(
            text=(
                f"Nodes searched: "
                f"{self.ai.nodes_searched:,}\n"
                f"Branches pruned: "
                f"{self.ai.pruned_branches:,}\n"
                f"Thinking time: "
                f"{self.ai.last_thinking_time:.3f}s"
            )
        )

    # ========================================================
    # END GAME + PLAY AGAIN
    # ========================================================

    def end_game(self):

        # Prevent duplicate dialogs caused by repeated callbacks
        if self.game_over_shown:
            return

        self.game_over_shown = True
        self.ai_thinking = False

        black = self.game.count_discs(
            BLACK
        )

        white = self.game.count_discs(
            WHITE
        )

        if black > white:

            winner = "BLACK"
            result_message = "You Win!"

        elif white > black:

            winner = "WHITE / AI"
            result_message = "AI Wins!"

        else:

            winner = "DRAW"
            result_message = "It's a Draw!"

        self.ai_status.config(
            text="GAME COMPLETE",
            fg=self.accent
        )

        self.turn_label.config(
            text="GAME OVER",
            fg=self.accent
        )

        self.status.config(
            text=(
                f"Final score — "
                f"Black: {black} | "
                f"White: {white} | "
                f"Result: {winner}"
            )
        )

        self.draw_board()
        self.update_information()

        # ----------------------------------------------------
        # PLAY AGAIN DIALOG
        # ----------------------------------------------------

        play_again = messagebox.askyesno(
            "Game Over - Play Again?",
            (
                f"{result_message}\n\n"
                f"FINAL SCORE\n"
                f"Black: {black}\n"
                f"White: {white}\n\n"
                f"Winner: {winner}\n\n"
                f"Would you like to play again?"
            )
        )

        if play_again:

            self.new_game()

        else:

            self.status.config(
                text=(
                    f"Game complete — "
                    f"Black: {black} | "
                    f"White: {white} | "
                    f"Winner: {winner}. "
                    f"Press NEW GAME to play again."
                )
            )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():

    root = tk.Tk()

    app = OthelloGUI(root)

    root.mainloop()


if __name__ == "__main__":
    main()
