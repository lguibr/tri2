# File: ui/input_handler.py
import pygame
from typing import Tuple, Callable, Dict, TYPE_CHECKING

# Type Aliases for Callbacks
HandleDemoMouseMotionCallback = Callable[[Tuple[int, int]], None]
HandleDemoMouseButtonDownCallback = Callable[[pygame.event.Event], None]
ToggleTrainingRunCallback = Callable[[], None]
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]
StartDebugModeCallback = Callable[[], None]
ExitDebugModeCallback = Callable[[], None]
HandleDebugInputCallback = Callable[[pygame.event.Event], None]

# Forward declaration for type hinting
if TYPE_CHECKING:
    from .renderer import UIRenderer
    from app_state import AppState  # Import Enum


class InputHandler:
    """Handles Pygame events and triggers callbacks based on application state."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        toggle_training_run_cb: ToggleTrainingRunCallback,
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_mouse_motion_cb: HandleDemoMouseMotionCallback,
        handle_demo_mouse_button_down_cb: HandleDemoMouseButtonDownCallback,
        start_debug_mode_cb: StartDebugModeCallback,
        exit_debug_mode_cb: ExitDebugModeCallback,
        handle_debug_input_cb: HandleDebugInputCallback,
    ):
        self.screen = screen
        self.renderer = renderer
        self.toggle_training_run_cb = toggle_training_run_cb
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

        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

        # Button rects are now managed within ButtonStatusRenderer,
        # but we need references for click detection here.
        self.run_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.cleanup_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.demo_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.debug_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_yes_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_no_rect = pygame.Rect(0, 0, 0, 0)
        self._update_button_rects()  # Initial calculation

    def _update_button_rects(self):
        """Calculates button rects based on initial layout assumptions."""
        # These rects are primarily for click detection, actual rendering is in LeftPanel
        # TODO: Get these rects dynamically from the ButtonStatusRenderer if possible
        # For now, keep the static calculation as a fallback.
        button_height = 40
        button_y_pos = 10
        run_button_width = 100
        cleanup_button_width = 160
        demo_button_width = 120
        debug_button_width = 120
        button_spacing = 10

        self.run_btn_rect = pygame.Rect(
            button_spacing, button_y_pos, run_button_width, button_height
        )
        current_x = self.run_btn_rect.right + button_spacing
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

        # Confirmation buttons are positioned dynamically during rendering
        sw, sh = self.screen.get_size()
        self.confirm_yes_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_no_rect = pygame.Rect(0, 0, 100, 40)
        self.confirm_yes_rect.center = (
            sw // 2 - 60,
            sh // 2 + 50,
        )  # Approximate center
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)  # Approximate center

    def handle_input(
        self, app_state_str: str, cleanup_confirmation_active: bool
    ) -> bool:
        """
        Processes Pygame events. Returns True to continue running, False to exit.
        """
        from app_state import AppState  # Local import for Enum comparison

        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)

        # Update dynamic rects (like confirmation buttons)
        sw, sh = self.screen.get_size()
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

        # Update shape preview rects if in demo mode
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

        if (
            app_state_str == AppState.MAIN_MENU.value
            and not cleanup_confirmation_active
        ):
            if hasattr(self.renderer, "check_hover"):
                self.renderer.check_hover(mouse_pos, app_state_str)

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
                    self._update_button_rects()  # Recalculate static button rects on resize
                    if hasattr(self.renderer, "force_redraw"):
                        self.renderer.force_redraw()
                    print(f"Window resized: {new_w}x{new_h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
                continue

            # --- Cleanup Confirmation Mode ---
            if cleanup_confirmation_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.cancel_cleanup_cb()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.confirm_yes_rect.collidepoint(mouse_pos):
                        self.confirm_cleanup_cb()
                    elif self.confirm_no_rect.collidepoint(mouse_pos):
                        self.cancel_cleanup_cb()
                continue

            # --- Playing (Demo) Mode ---
            elif app_state_str == AppState.PLAYING.value:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_demo_mode_cb()
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_demo_mouse_motion_cb(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_demo_mouse_button_down_cb(event)

            # --- Debug Mode ---
            elif app_state_str == AppState.DEBUG.value:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_debug_mode_cb()
                    else:
                        self.handle_debug_input_cb(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_debug_input_cb(event)

            # --- Main Menu Mode ---
            elif app_state_str == AppState.MAIN_MENU.value:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return self.exit_app_cb()
                    elif event.key == pygame.K_p:
                        self.toggle_training_run_cb()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Use the stored rects for collision detection
                    if self.run_btn_rect.collidepoint(mouse_pos):
                        self.toggle_training_run_cb()
                    elif self.cleanup_btn_rect.collidepoint(mouse_pos):
                        self.request_cleanup_cb()
                    elif self.demo_btn_rect.collidepoint(mouse_pos):
                        self.start_demo_mode_cb()
                    elif self.debug_btn_rect.collidepoint(mouse_pos):
                        self.start_debug_mode_cb()

            # --- Error Mode ---
            elif app_state_str == AppState.ERROR.value:
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
            getattr(self.renderer, "tooltips", None),
            getattr(self.renderer, "demo_renderer", None),
            # Add demo sub-renderers if they hold screen refs
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
