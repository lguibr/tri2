# File: ui/input_handler.py
# (No significant changes needed, this file was already reasonably focused)
import pygame
from typing import Tuple, Callable, Optional

# --- Define Callback Types (for clarity) ---
HandleDemoInputCallback = Callable[[pygame.event.Event], None]
ToggleTrainingCallback = Callable[[], None]
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]  # Returns False to signal exit
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]
# --- End Callback Types ---

# Forward declaration for type hinting UIRenderer
if False:  # Prevent circular import at runtime
    from .renderer import UIRenderer


class InputHandler:
    """Handles Pygame events and triggers callbacks based on application state."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",  # Use forward declaration
        toggle_training_cb: ToggleTrainingCallback,
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_input_cb: HandleDemoInputCallback,
    ):
        self.screen = screen  # Keep reference to update size on resize
        self.renderer = renderer
        # Store callbacks provided by MainApp
        self.toggle_training_cb = toggle_training_cb
        self.request_cleanup_cb = request_cleanup_cb
        self.cancel_cleanup_cb = cancel_cleanup_cb
        self.confirm_cleanup_cb = confirm_cleanup_cb
        self.exit_app_cb = exit_app_cb
        self.start_demo_mode_cb = start_demo_mode_cb
        self.exit_demo_mode_cb = exit_demo_mode_cb
        self.handle_demo_input_cb = handle_demo_input_cb

        # Store button rect definitions locally for collision detection
        # Note: These are visually defined in LeftPanelRenderer, coupling exists.
        self._update_button_rects()  # Initialize rects

    def _update_button_rects(self):
        """Calculates button rects based on initial layout assumptions."""
        # These might need adjustment if the layout in LeftPanelRenderer changes significantly.
        self.train_btn_rect = pygame.Rect(10, 10, 100, 40)
        self.cleanup_btn_rect = pygame.Rect(self.train_btn_rect.right + 10, 10, 160, 40)
        self.demo_btn_rect = pygame.Rect(self.cleanup_btn_rect.right + 10, 10, 120, 40)
        # Confirmation buttons are relative to screen center, calculated dynamically
        sw, sh = self.screen.get_size()
        self.confirm_yes_rect = pygame.Rect(sw // 2 - 110, sh // 2 + 30, 100, 40)
        self.confirm_no_rect = pygame.Rect(sw // 2 + 10, sh // 2 + 30, 100, 40)

    def handle_input(self, app_state: str, cleanup_confirmation_active: bool) -> bool:
        """
        Processes Pygame events based on the current app_state and cleanup overlay status.
        Returns True to continue running, False to exit.
        """
        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)  # Fallback if display not initialized/error

        # Update confirmation button rects in case of resize
        sw, sh = self.screen.get_size()
        self.confirm_yes_rect.center = (sw // 2 - 60, sh // 2 + 50)
        self.confirm_no_rect.center = (sw // 2 + 60, sh // 2 + 50)

        # Check tooltip hover (only if overlay is not active and in MainMenu)
        # Renderer handles tooltip display, input handler just triggers the check.
        if app_state == "MainMenu" and not cleanup_confirmation_active:
            if hasattr(self.renderer, "check_hover"):
                self.renderer.check_hover(mouse_pos, app_state)

        # --- Event Loop ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.exit_app_cb()  # Signal exit immediately

            # Handle Window Resizing
            if event.type == pygame.VIDEORESIZE:
                try:
                    # Ensure minimum size
                    new_w, new_h = max(320, event.w), max(240, event.h)
                    self.screen = pygame.display.set_mode(
                        (new_w, new_h), pygame.RESIZABLE
                    )
                    # Update screen references in all UI components that hold one
                    self._update_ui_screen_references(self.screen)
                    # Update local button rects that depend on screen size
                    self._update_button_rects()
                    # Force plotter redraw as its size changed
                    if hasattr(self.renderer, "force_redraw"):
                        self.renderer.force_redraw()
                    print(f"Window resized: {new_w}x{new_h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
                continue  # Skip other event processing for this frame after resize

            # --- State-Specific Input Handling ---

            # 1. Cleanup Confirmation Overlay (Highest Priority)
            if cleanup_confirmation_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.cancel_cleanup_cb()
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                ):  # Left click
                    if self.confirm_yes_rect.collidepoint(mouse_pos):
                        self.confirm_cleanup_cb()
                    elif self.confirm_no_rect.collidepoint(mouse_pos):
                        self.cancel_cleanup_cb()
                # Consume event if overlay is active, preventing other actions
                continue

            # 2. Playing State (Demo Mode)
            elif app_state == "Playing":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_demo_mode_cb()
                    else:
                        # Delegate game control input to specific handler
                        self.handle_demo_input_cb(event)
                # No other input handled in this state currently

            # 3. Main Menu State
            elif app_state == "MainMenu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return self.exit_app_cb()  # Exit App
                    elif event.key == pygame.K_p:
                        self.toggle_training_cb()  # Toggle Training

                if (
                    event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                ):  # Left click
                    if self.train_btn_rect.collidepoint(mouse_pos):
                        self.toggle_training_cb()
                    elif self.cleanup_btn_rect.collidepoint(mouse_pos):
                        self.request_cleanup_cb()  # Show confirmation
                    elif self.demo_btn_rect.collidepoint(mouse_pos):
                        self.start_demo_mode_cb()
                    # No action for clicking elsewhere in main menu

            # 4. Error State
            elif app_state == "Error":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return self.exit_app_cb()  # Allow exit from error state
                # No other input handled in error state

            # 5. Other States (e.g., Initializing, Cleaning) - Ignore input for now
            # else: pass

        return True  # Continue running

    def _update_ui_screen_references(self, new_screen: pygame.Surface):
        """Recursively updates the screen reference in the renderer and its sub-components."""
        # List of components known to hold a screen reference
        components_to_update = [
            self.renderer,
            getattr(self.renderer, "left_panel", None),
            getattr(self.renderer, "game_area", None),
            getattr(self.renderer, "overlays", None),
            getattr(self.renderer, "tooltips", None),
            getattr(
                self.renderer, "demo_renderer", None
            ),  # Update new demo renderer too
        ]
        for component in components_to_update:
            if component and hasattr(component, "screen"):
                component.screen = new_screen
