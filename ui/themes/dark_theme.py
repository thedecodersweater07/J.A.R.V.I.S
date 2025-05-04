class DarkTheme:
    def get_colors(self):
        return {
            "TEXT": (0.90, 0.90, 0.90, 1.0),
            "WINDOW_BACKGROUND": (0.05, 0.05, 0.05, 0.95),
            "TITLE_BACKGROUND_ACTIVE": (0.10, 0.10, 0.10, 1.0),
            "FRAME_BACKGROUND": (0.15, 0.15, 0.15, 1.0),
            "FRAME_BACKGROUND_HOVERED": (0.20, 0.20, 0.20, 1.0),
            "BUTTON": (0.15, 0.15, 0.15, 0.8),
            "BUTTON_HOVERED": (0.20, 0.20, 0.20, 1.0)
        }

    def get_metrics(self):
        return {
            "window_padding": (10, 10),
            "frame_padding": (5, 5),
            "item_spacing": (8, 4),
            "scrollbar_size": 13,
            "frame_rounding": 3,
            "grab_rounding": 3
        }
