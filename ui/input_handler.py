# File: ui/input_handler.py
# --- Pygame Input Handling Logic ---
import pygame
from typing import Tuple, Callable, Optional

# Define callback types for actions
ToggleTrainingCallback = Callable[[], None]
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]  # Returns False to signal exit


class InputHandler:
    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",  # Forward reference if needed
        toggle_training_cb: ToggleTrainingCallback,
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
    ):
        self.screen = screen
        self.renderer = renderer
        self.toggle_training_cb = toggle_training_cb
        self.request_cleanup_cb = request_cleanup_cb
        self.cancel_cleanup_cb = cancel_cleanup_cb
        self.confirm_cleanup_cb = confirm_cleanup_cb
        self.exit_app_cb = exit_app_cb

    def handle_input(self, cleanup_confirmation_active: bool) -> bool:
        """Processes Pygame events and calls appropriate callbacks. Returns False to exit."""
        mouse_pos = pygame.mouse.get_pos()
        sw, sh = self.screen.get_size()

        # Define button rects (consider moving definitions to renderer or config)
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 160, 40)
        confirm_yes_rect = pygame.Rect(sw // 2 - 110, sh // 2 + 30, 100, 40)
        confirm_no_rect = pygame.Rect(sw // 2 + 10, sh // 2 + 30, 100, 40)

        # Check for tooltip hover (delegated to renderer)
        if hasattr(self.renderer, "check_hover"):
            self.renderer.check_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.exit_app_cb()  # Signal exit

            if event.type == pygame.VIDEORESIZE:
                try:
                    self.screen = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE
                    )
                    if hasattr(self.renderer, "screen"):
                        self.renderer.screen = self.screen
                    if hasattr(
                        self.renderer, "force_redraw"
                    ):  # Optional: signal renderer
                        self.renderer.force_redraw()
                    print(f"Window resized: {event.w}x{event.h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if cleanup_confirmation_active:
                        self.cancel_cleanup_cb()
                    else:
                        return self.exit_app_cb()  # Signal exit
                elif event.key == pygame.K_p and not cleanup_confirmation_active:
                    self.toggle_training_cb()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                if cleanup_confirmation_active:
                    if confirm_yes_rect.collidepoint(mouse_pos):
                        self.confirm_cleanup_cb()
                    elif confirm_no_rect.collidepoint(mouse_pos):
                        self.cancel_cleanup_cb()
                else:
                    if train_btn_rect.collidepoint(mouse_pos):
                        self.toggle_training_cb()
                    elif cleanup_btn_rect.collidepoint(mouse_pos):
                        self.request_cleanup_cb()
        return True  # Continue running
