# File: ui/input_handler.py
import pygame
from typing import Tuple, Callable, Optional

# --- NEW: Define Demo Input Callback ---
HandleDemoInputCallback = Callable[[pygame.event.Event], None]

# Modify callback types to include demo mode switching
ToggleTrainingCallback = Callable[[], None]
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]
# --- NEW: Callbacks for Demo Mode ---
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]
# --- END NEW ---


class InputHandler:
    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        toggle_training_cb: ToggleTrainingCallback,
        request_cleanup_cb: RequestCleanupCallback,
        cancel_cleanup_cb: CancelCleanupCallback,
        confirm_cleanup_cb: ConfirmCleanupCallback,
        exit_app_cb: ExitAppCallback,
        # --- NEW: Add Demo Callbacks ---
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_input_cb: HandleDemoInputCallback,
        # --- END NEW ---
    ):
        self.screen = screen
        self.renderer = renderer
        self.toggle_training_cb = toggle_training_cb
        self.request_cleanup_cb = request_cleanup_cb
        self.cancel_cleanup_cb = cancel_cleanup_cb
        self.confirm_cleanup_cb = confirm_cleanup_cb
        self.exit_app_cb = exit_app_cb
        # --- NEW: Store Demo Callbacks ---
        self.start_demo_mode_cb = start_demo_mode_cb
        self.exit_demo_mode_cb = exit_demo_mode_cb
        self.handle_demo_input_cb = handle_demo_input_cb
        # --- END NEW ---

    def handle_input(
        self, app_state: str, cleanup_confirmation_active: bool
    ) -> bool:  # Pass app_state
        """Processes Pygame events based on app_state. Returns False to exit."""
        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:  # Can happen if display context is lost briefly
            mouse_pos = (0, 0)

        sw, sh = self.screen.get_size()

        # Define button rects (consider moving definitions to renderer or config)
        # Rects for Main Menu
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 160, 40)
        demo_btn_rect = pygame.Rect(cleanup_btn_rect.right + 10, 10, 120, 40)  # NEW

        # Rects for Cleanup Confirmation (independent of app_state)
        confirm_yes_rect = pygame.Rect(sw // 2 - 110, sh // 2 + 30, 100, 40)
        confirm_no_rect = pygame.Rect(sw // 2 + 10, sh // 2 + 30, 100, 40)

        # Check for tooltip hover (now state-dependent in renderer)
        if hasattr(self.renderer, "check_hover"):
            self.renderer.check_hover(mouse_pos, app_state)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.exit_app_cb()  # Signal exit

            # Handle resizing universally
            if event.type == pygame.VIDEORESIZE:
                try:
                    # Basic sanity check on resize dimensions
                    new_w, new_h = max(320, event.w), max(240, event.h)
                    self.screen = pygame.display.set_mode(
                        (new_w, new_h), pygame.RESIZABLE
                    )
                    # Update renderer's screen reference
                    if hasattr(self.renderer, "screen"):
                        self.renderer.screen = self.screen
                    # Update panel screen references if they store it separately
                    if hasattr(self.renderer, "left_panel") and hasattr(
                        self.renderer.left_panel, "screen"
                    ):
                        self.renderer.left_panel.screen = self.screen
                    if hasattr(self.renderer, "game_area") and hasattr(
                        self.renderer.game_area, "screen"
                    ):
                        self.renderer.game_area.screen = self.screen
                    if hasattr(self.renderer, "overlays") and hasattr(
                        self.renderer.overlays, "screen"
                    ):
                        self.renderer.overlays.screen = self.screen
                    if hasattr(self.renderer, "tooltips") and hasattr(
                        self.renderer.tooltips, "screen"
                    ):
                        self.renderer.tooltips.screen = self.screen

                    if hasattr(self.renderer, "force_redraw"):
                        self.renderer.force_redraw()
                    print(f"Window resized: {new_w}x{new_h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")

            # --- State-Dependent Input Handling ---

            # 1. Cleanup Confirmation Overlay (takes precedence over app state)
            if cleanup_confirmation_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.cancel_cleanup_cb()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if confirm_yes_rect.collidepoint(mouse_pos):
                        self.confirm_cleanup_cb()
                    elif confirm_no_rect.collidepoint(mouse_pos):
                        self.cancel_cleanup_cb()
                # Don't process other inputs if confirmation is active
                # Use 'continue' to skip to the next event in the loop
                continue

            # 2. Playing State Input
            elif app_state == "Playing":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.exit_demo_mode_cb()  # Exit demo mode
                    else:
                        # Pass other keydown events to the demo handler
                        self.handle_demo_input_cb(event)
                # Add mouse handling for demo if needed (e.g., clicking UI elements)

            # 3. Main Menu State Input
            elif app_state == "MainMenu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return self.exit_app_cb()  # Exit app from main menu
                    elif event.key == pygame.K_p:
                        self.toggle_training_cb()

                if (
                    event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                ):  # Left click
                    # Check button collisions only if visible in MainMenu
                    if train_btn_rect.collidepoint(mouse_pos):
                        self.toggle_training_cb()
                    elif cleanup_btn_rect.collidepoint(mouse_pos):
                        self.request_cleanup_cb()
                    elif demo_btn_rect.collidepoint(
                        mouse_pos
                    ):  # Handle demo button click
                        self.start_demo_mode_cb()

            # 4. Error State Input
            elif app_state == "Error":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return self.exit_app_cb()  # Allow exit from error state

            # Handle other states (e.g., "Initializing") if needed
            # else: pass # No specific input handling for other states for now

        return True  # Continue running
