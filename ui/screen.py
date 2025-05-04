import glfw
import OpenGL.GL as gl
from OpenGL.GL import shaders
import imgui
from imgui.integrations.glfw import GlfwRenderer
import logging
from typing import Optional, Callable
import traceback
import signal
import numpy as np
from math import sin, cos
import time
from ui.themes.stark_theme import StarkTheme
from ui.themes.minimal_theme import MinimalTheme
from ui.themes.dark_theme import DarkTheme
from ui.themes.light_theme import LightTheme

class Screen:
    def __init__(self, width: int = 800, height: int = 600, title: str = "JARVIS Interface"):
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        self.title = title
        self.window = None
        self.impl = None
        self.input_text = ""
        self.chat_history = []
        self.callback = None
        self.error_state = False
        self.last_error = None
        self.is_initialized = False
        self.running = False
        self.interrupt_received = False
        self.theme = StarkTheme()  # Default theme
        self.loading = True
        self.loading_angle = 0.0
        self.last_frame_time = time.time()
        self.available_themes = {
            "Stark": StarkTheme(),
            "Minimal": MinimalTheme(),
            "Dark": DarkTheme(),
            "Light": LightTheme()
        }
        self.current_theme_name = "Stark"
        self.current_theme = self.available_themes[self.current_theme_name]
        self.show_theme_selector = False
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.imgui_initialized = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully"""
        self.interrupt_received = True
        if self.window:
            glfw.set_window_should_close(self.window, True)
        self.logger.info("Interrupt received, initiating graceful shutdown...")

    def init(self):
        """Initialize screen with modern OpenGL"""
        try:
            if not glfw.init():
                self.logger.error("Could not initialize GLFW")
                return False

            # Request OpenGL 3.3 core profile
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
            glfw.window_hint(glfw.SAMPLES, 4)

            self.window = glfw.create_window(
                self.width, self.height, self.title, None, None
            )
            if not self.window:
                self.logger.error("Failed to create GLFW window")
                glfw.terminate()
                return False

            glfw.make_context_current(self.window)

            # Check OpenGL version
            version = gl.glGetString(gl.GL_VERSION).decode('utf-8')
            self.logger.info(f"OpenGL Version: {version}")

            # Basic OpenGL setup
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_MULTISAMPLE)

            # Setup modern OpenGL
            if self._setup_modern_gl():
                # Initialize ImGui
                imgui.create_context()
                self.impl = GlfwRenderer(self.window)
                self.imgui_initialized = True

                # Setup ImGui style
                self._setup_style()
                self._setup_3d_loading()

                self.is_initialized = True
                self.logger.info("Screen initialized successfully")
                return True
            return False

        except Exception as e:
            self.error_state = True
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize screen: {e}\n{traceback.format_exc()}")
            if glfw.get_current_context():
                glfw.terminate()
            return False

    def _setup_modern_gl(self):
        """Setup modern OpenGL shaders and buffers"""
        try:
            vertex_shader = """
            #version 330 core
            layout(location = 0) in vec3 position;
            uniform mat4 model;
            void main() {
                gl_Position = model * vec4(position, 1.0);
            }
            """

            fragment_shader = """
            #version 330 core
            out vec4 fragColor;
            void main() {
                fragColor = vec4(0.2, 0.5, 0.8, 1.0);
            }
            """

            # Compile and link shaders
            vert = shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
            self.shader_program = shaders.compileProgram(vert, frag)

            # Create triangle vertices
            vertices = np.array([
                -0.6, -0.6, 0.0,
                 0.6, -0.6, 0.0,
                 0.0,  0.6, 0.0
            ], dtype=np.float32)

            # Generate and bind VAO first
            if gl.glGenVertexArrays and gl.glBindVertexArray:
                self.vao = gl.glGenVertexArrays(1)
                gl.glBindVertexArray(self.vao)
            else:
                self.logger.warning("VAOs not supported, falling back to legacy OpenGL")
                return False

            # Generate and bind VBO
            self.vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

            # Setup vertex attributes
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(0)

            # Cleanup state
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindVertexArray(0)

            return True

        except Exception as e:
            self.logger.error(f"Error setting up modern OpenGL: {e}")
            return False

    def _setup_3d_loading(self):
        """Setup modern OpenGL 3D loading screen"""
        try:
            # Basic shaders
            vertex_shader = """
                #version 330 core
                layout(location = 0) in vec3 position;
                uniform mat4 model;
                void main() {
                    gl_Position = model * vec4(position, 1.0);
                }
            """
            fragment_shader = """
                #version 330 core
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(0.2, 0.5, 0.8, 1.0);
                }
            """

            # Compile shaders
            vert = shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
            self.shader = shaders.compileProgram(vert, frag)

            # Create loading animation geometry
            vertices = np.array([
                -0.6, -0.6, 0.0,
                 0.6, -0.6, 0.0,
                 0.0,  0.6, 0.0
            ], dtype=np.float32)

            # Create and bind VAO & VBO if supported
            if hasattr(gl, 'glGenVertexArrays'):
                self.vao = gl.glGenVertexArrays(1)
                gl.glBindVertexArray(self.vao)

                self.vbo = gl.glGenBuffers(1)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

                gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
                gl.glEnableVertexAttribArray(0)

                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                gl.glBindVertexArray(0)

        except Exception as e:
            self.logger.error(f"Error setting up 3D loading: {e}")
            raise

    def cleanup(self):
        """Cleanup OpenGL resources safely"""
        try:
            if self.impl:
                self.impl.shutdown()

            if hasattr(gl, 'glDeleteVertexArrays') and self.vao:
                gl.glDeleteVertexArrays([self.vao])

            if hasattr(gl, 'glDeleteBuffers') and self.vbo:
                gl.glDeleteBuffers([self.vbo])

            if hasattr(self, 'shader') and self.shader:
                gl.glDeleteProgram(self.shader)

            if self.window:
                glfw.destroy_window(self.window)
                self.window = None

            glfw.terminate()
            self.logger.info("Screen cleanup completed successfully")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def set_theme(self, theme_name: str):
        """Change current theme"""
        if theme_name in self.available_themes:
            self.current_theme_name = theme_name
            self.current_theme = self.available_themes[theme_name]
            self._apply_theme(self.current_theme)
            self.logger.info(f"Changed theme to {theme_name}")

    def _apply_theme(self, theme):
        """Apply theme settings"""
        style = imgui.get_style()
        
        colors = theme.get_colors()
        for name, color in colors.items():
            style.colors[getattr(imgui, f"COLOR_{name}")] = color
            
        metrics = theme.get_metrics()
        for name, value in metrics.items():
            setattr(style, name, value)

    def _setup_style(self):
        """Apply current theme styles"""
        self._apply_theme(self.theme)

    def _render_theme_selector(self):
        """Render theme selection menu"""
        if imgui.begin_menu("Themes"):
            for theme_name in self.available_themes.keys():
                selected = self.current_theme_name == theme_name
                if imgui.menu_item(theme_name, None, selected)[0]:
                    self.set_theme(theme_name)
            imgui.end_menu()

    def _handle_resize(self, window, width, height):
        self.width = width
        self.height = height
        gl.glViewport(0, 0, width, height)

    def _handle_keyboard(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def render(self):
        if not self.is_initialized or not self.window:
            return

        try:
            if glfw.window_should_close(self.window):
                return

            glfw.poll_events()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            if self.imgui_initialized and self.impl:
                self.impl.process_inputs()
                imgui.new_frame()
                
                self._render_main_window()
                
                imgui.render()
                self.impl.render(imgui.get_draw_data())
            
            glfw.swap_buffers(self.window)
            
        except Exception as e:
            self.error_state = True
            self.last_error = str(e)
            self.logger.error(f"Render error: {e}\n{traceback.format_exc()}")

    def _render_main_window(self):
        """Render main application window"""
        imgui.set_next_window_size(self.width, self.height)
        imgui.set_next_window_position(0, 0)
        
        imgui.begin(
            "JARVIS Chat",
            flags=(
                imgui.WINDOW_NO_TITLE_BAR |
                imgui.WINDOW_NO_RESIZE |
                imgui.WINDOW_NO_MOVE
            )
        )
        
        self._render_chat_history()
        self._handle_input()
        
        imgui.end()

    def _render_chat_history(self):
        try:
            # Calculate chat window height (leaving space for input)
            chat_height = self.height - 100
            
            imgui.begin_child(
                "ScrollRegion",
                width=0,
                height=chat_height,
                border=True
            )
            
            for msg, is_user in self.chat_history:
                if is_user:
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.2, 0.7, 0.2)
                    imgui.text_wrapped(f"You: {msg}")
                else:
                    imgui.push_style_color(imgui.COLOR_TEXT, 0.2, 0.2, 0.8)
                    imgui.text_wrapped(f"JARVIS: {msg}")
                imgui.pop_style_color()
                imgui.spacing()
            
            # Auto-scroll to bottom
            if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
                imgui.set_scroll_here_y(1.0)
            
            imgui.end_child()
            
        except Exception as e:
            self.logger.error(f"Error rendering chat history: {e}")
            raise

    def _handle_input(self):
        try:
            # Input text field
            imgui.push_item_width(self.width - 120)  # Leave space for button
            changed, self.input_text = imgui.input_text(
                "##Input",
                self.input_text,
                256,
                imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
            )
            imgui.pop_item_width()
            
            # Send button
            imgui.same_line()
            if imgui.button("Send") or changed:
                if self.input_text.strip() and self.callback:
                    self.callback(self.input_text.strip())
                    self.input_text = ""
                    
        except Exception as e:
            self.logger.error(f"Error handling input: {e}")
            raise

    def add_message(self, text: str, is_user: bool = False):
        try:
            if not isinstance(text, str):
                text = str(text)
            self.chat_history.append((text, is_user))
        except Exception as e:
            self.logger.error(f"Error adding message: {e}")

    def set_callback(self, callback: Callable[[str], None]):
        self.callback = callback

    def should_close(self) -> bool:
        return bool(self.window and glfw.window_should_close(self.window))
