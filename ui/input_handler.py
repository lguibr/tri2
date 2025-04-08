# File: ui/input_handler.py
# File: ui/input_handler.py
import pygame
from typing import Tuple, Callable, Dict, TYPE_CHECKING, Optional

# Type Aliases for Callbacks
HandleDemoMouseMotionCallback = Callable[[Tuple[int, int]], None]
HandleDemoMouseButtonDownCallback = Callable[[pygame.event.Event], None]
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]
StartDebugModeCallback = Callable[[], None]
ExitDebugModeCallback = Callable[[], None]
HandleDebugInputCallback = Callable[[pygame.event.Event], None]
# MCTS Vis Callbacks Removed
# StartMCTSVisualizationCallback = Callable[[], None]
# ExitMCTSVisualizationCallback = Callable[[], None]
# HandleMCTSPanCallback = Callable[[int, int], None]
# HandleMCTSZoomCallback = Callable[[float, Tuple[int, int]], None]
# Combined Worker Control Callbacks
StartRunCallback = Callable[[], None]
StopRunCallback = Callable[[], None]


if TYPE_CHECKING:
    from .renderer import UIRenderer
    from app_state import AppState
    from main_pygame import MainApp


class InputHandler:
    """Handles Pygame events and triggers callbacks based on application state."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        # Basic Callbacks
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
        # Demo Mode Callbacks
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_mouse_motion_cb: HandleDemoMouseMotionCallback,
        handle_demo_mouse_button_down_cb: HandleDemoMouseButtonDownCallback,
        # Debug Mode Callbacks
        start_debug_mode_cb: StartDebugModeCallback,
        exit_debug_mode_cb: ExitDebugModeCallback,
        handle_debug_input_cb: HandleDebugInputCallback,
        # MCTS Vis Callbacks Removed
        # start_mcts_visualization_cb: StartMCTSVisualizationCallback,
        # exit_mcts_visualization_cb: ExitMCTSVisualizationCallback,
        # handle_mcts_pan_cb: HandleMCTSPanCallback,
        # handle_mcts_zoom_cb: HandleMCTSZoomCallback,
        # Combined Worker Control Callbacks
        start_run_cb: StartRunCallback,
        stop_run_cb: StopRunCallback,
    ):
        self.screen = screen
        self.renderer = renderer
        # Store Callbacks
        self.request_cleanup_cb = request_cleanup_cb
        self.cancel_cleanup_cb = cancel_cleanup_cb
        self.confirm_cleanup_cb = confirm_cleanup_cb
        self.exit_app_cb = exit_app_cb
        self.start_demo_mode_cb = start_demo_mode_cb
        self.exit_demo_mode_cb = exit_demo_mode_cb
        self.handle_demo_mouse_motion_cb = handle_demo_mouse_motion_cb
        self.handle_demo_mouse_button_down_cb = handle_demo_mouse_button_down_cb
        self.start_debug_mode_cb = start_debug_mode_cb
        self.exit_debug_mode_cb = exit_debug_mode_cb
        self.handle_debug_input_cb = handle_debug_input_cb
        # MCTS Vis Callbacks Removed
        # self.start_mcts_visualization_cb = start_mcts_visualization_cb
        # self.exit_mcts_visualization_cb = exit_mcts_visualization_cb
        # self.handle_mcts_pan_cb = handle_mcts_pan_cb
        # self.handle_mcts_zoom_cb = handle_mcts_zoom_cb
        self.start_run_cb = start_run_cb
        self.stop_run_cb = stop_run_cb

        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

        # Button rects
        self.run_stop_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.cleanup_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.demo_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.debug_btn_rect = pygame.Rect(0, 0, 0, 0)
        # self.mcts_vis_btn_rect = pygame.Rect(0, 0, 0, 0) # MCTS Vis removed
        self.confirm_yes_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_no_rect = pygame.Rect(0, 0, 0, 0)
        self._update_button_rects()

        # MCTS Vis state removed
        # self.is_panning_mcts = False
        # self.last_pan_pos: Optional[Tuple[int, int]] = None
        self.app_ref: Optional["MainApp"] = None

    def _update_button_rects(self):
        """Calculates button rects based on initial layout assumptions."""
        button_height = 40
        button_y_pos = 10
        run_stop_button_width = 150
        cleanup_button_width = 160
        demo_button_width = 120
        debug_button_width = 120
        # mcts_vis_button_width = 140 # MCTS Vis removed
        button_spacing = 10

        current_x = button_spacing
        self.run_stop_btn_rect = pygame.Rect(
            current_x, button_y_pos, run_stop_button_width, button_height
        )
        current_x = self.run_stop_btn_rect.right + button_spacing * 2
        self.cleanup_btn_rect = pygame.Rect(
            current_x, button_y_pos, cleanup_button_width, button_height
        )
        current_x = self.cleanup_btn_rect.right + button_spacing
        self.demo_btn_rect = pygame.Rect(
            current_x, button_y_pos, demo_button_width, button_height
        )
        current_x = self.demo_btn_rect.right + button_spacing
        self.debug_btn_rect = pygame.Rect(
            current_x, button_y_pos, debug_button_width, button_height
        )
        sw, sh = self.screen.get_size()
        self.confirm_yes_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_no_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

    def handle_input(
        self, app_state_str: str, cleanup_confirmation_active: bool
    ) -> bool:
        """Processes Pygame events. Returns True to continue running, False to exit."""
        from app_state import AppState

        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)

        sw, sh = self.screen.get_size()
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

        if (
            app_state_str == AppState.PLAYING.value
            and self.renderer
            and self.renderer.demo_renderer
        ):
            self.shape_preview_rects = (
                self.renderer.demo_renderer.get_shape_preview_rects()
            )
        else:
            self.shape_preview_rects.clear()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.exit_app_cb()

            if event.type == pygame.VIDEORESIZE:
                try:
                    new_w, new_h = max(320, event.w), max(240, event.h)
                    self.screen = pygame.display.set_mode(
                        (new_w, new_h), pygame.RESIZABLE
                    )
                    self._update_ui_screen_references(self.screen)
                    self._update_button_rects()
                    if hasattr(self.renderer, "force_redraw"):
                        self.renderer.force_redraw()
                    print(f"Window resized: {new_w}x{new_h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
                continue

            if cleanup_confirmation_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.cancel_cleanup_cb()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.confirm_yes_rect.collidepoint(mouse_pos):
                        self.confirm_cleanup_cb()
                    elif self.confirm_no_rect.collidepoint(mouse_pos):
                        self.cancel_cleanup_cb()
                continue

            current_app_state = (
                AppState(app_state_str)
                if app_state_str in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            if current_app_state == AppState.PLAYING:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.exit_demo_mode_cb()
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_demo_mouse_motion_cb(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_demo_mouse_button_down_cb(event)

            elif current_app_state == AppState.DEBUG:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_debug_mode_cb()
                    else:
                        self.handle_debug_input_cb(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_debug_input_cb(event)

            elif current_app_state == AppState.MAIN_MENU:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return self.exit_app_cb()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    is_running = (
                        self.app_ref.worker_manager.is_any_worker_running()
                        if self.app_ref
                        else False
                    )

                    if self.run_stop_btn_rect.collidepoint(mouse_pos):
                        if is_running:
                            self.stop_run_cb()
                        else:
                            self.start_run_cb()
                    elif not is_running:  # Only allow other buttons if not running
                        if self.cleanup_btn_rect.collidepoint(mouse_pos):
                            self.request_cleanup_cb()
                        elif self.demo_btn_rect.collidepoint(mouse_pos):
                            self.start_demo_mode_cb()
                        elif self.debug_btn_rect.collidepoint(mouse_pos):
                            self.start_debug_mode_cb()

            elif current_app_state == AppState.ERROR:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return self.exit_app_cb()

        return True

    def _update_ui_screen_references(self, new_screen: pygame.Surface):
        """Updates the screen reference in the renderer and its sub-components."""
        components_to_update = [
            self.renderer,
            getattr(self.renderer, "left_panel", None),
            getattr(self.renderer, "game_area", None),
            getattr(self.renderer, "overlays", None),
            getattr(self.renderer, "demo_renderer", None),
            getattr(
                getattr(self.renderer, "demo_renderer", None), "grid_renderer", None
            ),
            getattr(
                getattr(self.renderer, "demo_renderer", None), "preview_renderer", None
            ),
            getattr(
                getattr(self.renderer, "demo_renderer", None), "hud_renderer", None
            ),
        ]
        for component in components_to_update:
            if component and hasattr(component, "screen"):
                component.screen = new_screen
