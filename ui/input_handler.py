# File: ui/input_handler.py
import pygame
from typing import Tuple, Callable, Optional

HandleDemoInputCallback = Callable[[pygame.event.Event], None]
ToggleRunningCallback = Callable[[], None]  # Renamed
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]

if False:
    from .renderer import UIRenderer


class InputHandler:
    """Handles Pygame events and triggers callbacks based on application state."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        toggle_running_cb: ToggleRunningCallback,  # Renamed
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_input_cb: HandleDemoInputCallback,
    ):
        self.screen = screen
        self.renderer = renderer
        self.toggle_running_cb = toggle_running_cb  # Renamed
        self.request_cleanup_cb = request_cleanup_cb
        self.cancel_cleanup_cb = cancel_cleanup_cb
        self.confirm_cleanup_cb = confirm_cleanup_cb
        self.exit_app_cb = exit_app_cb
        self.start_demo_mode_cb = start_demo_mode_cb
        self.exit_demo_mode_cb = exit_demo_mode_cb
        self.handle_demo_input_cb = handle_demo_input_cb

        self._update_button_rects()

    def _update_button_rects(self):
        """Calculates button rects based on initial layout assumptions."""
        self.run_btn_rect = pygame.Rect(10, 10, 100, 40)  # Renamed from train_btn_rect
        self.cleanup_btn_rect = pygame.Rect(self.run_btn_rect.right + 10, 10, 160, 40)
        self.demo_btn_rect = pygame.Rect(self.cleanup_btn_rect.right + 10, 10, 120, 40)
        sw, sh = self.screen.get_size()
        self.confirm_yes_rect = pygame.Rect(sw // 2 - 110, sh // 2 + 30, 100, 40)
        self.confirm_no_rect = pygame.Rect(sw // 2 + 10, sh // 2 + 30, 100, 40)

    def handle_input(self, app_state: str, cleanup_confirmation_active: bool) -> bool:
        """
        Processes Pygame events. Returns True to continue running, False to exit.
        """
        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)

        sw, sh = self.screen.get_size()
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

        if app_state == "MainMenu" and not cleanup_confirmation_active:
            if hasattr(self.renderer, "check_hover"):
                self.renderer.check_hover(mouse_pos, app_state)

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

            elif app_state == "Playing":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_demo_mode_cb()
                    else:
                        self.handle_demo_input_cb(event)

            elif app_state == "MainMenu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return self.exit_app_cb()
                    elif event.key == pygame.K_p:
                        self.toggle_running_cb()  # Toggle Run

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.run_btn_rect.collidepoint(mouse_pos):  # Use run_btn_rect
                        self.toggle_running_cb()
                    elif self.cleanup_btn_rect.collidepoint(mouse_pos):
                        self.request_cleanup_cb()
                    elif self.demo_btn_rect.collidepoint(mouse_pos):
                        self.start_demo_mode_cb()

            elif app_state == "Error":
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
        ]
        for component in components_to_update:
            if component and hasattr(component, "screen"):
                component.screen = new_screen
