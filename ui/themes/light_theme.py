class LightTheme:
    def get_colors(self):
        return {
            "TEXT": (0.10, 0.10, 0.10, 1.0),
            "WINDOW_BACKGROUND": (0.95, 0.95, 0.95, 0.95),
            "TITLE_BACKGROUND_ACTIVE": (0.90, 0.90, 0.90, 1.0),
            "FRAME_BACKGROUND": (0.85, 0.85, 0.85, 1.0),
            "FRAME_BACKGROUND_HOVERED": (0.80, 0.80, 0.80, 1.0),
            "BUTTON": (0.80, 0.80, 0.80, 0.8),
            "BUTTON_HOVERED": (0.75, 0.75, 0.75, 1.0)
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
