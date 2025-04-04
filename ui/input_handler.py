# File: ui/input_handler.py
import pygame
from typing import Tuple, Callable, Optional

# Define callbacks
HandleDemoInputCallback = Callable[[pygame.event.Event], None]
ToggleTrainingCallback = Callable[[], None]
RequestCleanupCallback = Callable[[], None]
CancelCleanupCallback = Callable[[], None]
ConfirmCleanupCallback = Callable[[], None]
ExitAppCallback = Callable[[], bool]
StartDemoModeCallback = Callable[[], None]
ExitDemoModeCallback = Callable[[], None]


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
        start_demo_mode_cb: StartDemoModeCallback,
        exit_demo_mode_cb: ExitDemoModeCallback,
        handle_demo_input_cb: HandleDemoInputCallback,
    ):
        self.screen = screen
        self.renderer = renderer
        self.toggle_training_cb = toggle_training_cb
        self.request_cleanup_cb = request_cleanup_cb
        self.cancel_cleanup_cb = cancel_cleanup_cb
        self.confirm_cleanup_cb = confirm_cleanup_cb
        self.exit_app_cb = exit_app_cb
        self.start_demo_mode_cb = start_demo_mode_cb
        self.exit_demo_mode_cb = exit_demo_mode_cb
        self.handle_demo_input_cb = handle_demo_input_cb

    def handle_input(self, app_state: str, cleanup_confirmation_active: bool) -> bool:
        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)

        sw, sh = self.screen.get_size()

        # Define button rects (consider moving)
        train_btn_rect = pygame.Rect(10, 10, 100, 40)
        cleanup_btn_rect = pygame.Rect(train_btn_rect.right + 10, 10, 160, 40)
        demo_btn_rect = pygame.Rect(cleanup_btn_rect.right + 10, 10, 120, 40)
        confirm_yes_rect = pygame.Rect(sw // 2 - 110, sh // 2 + 30, 100, 40)
        confirm_no_rect = pygame.Rect(sw // 2 + 10, sh // 2 + 30, 100, 40)

        # Check tooltip hover
        if hasattr(self.renderer, "check_hover"):
            self.renderer.check_hover(mouse_pos, app_state)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.exit_app_cb()

            # Handle resizing
            if event.type == pygame.VIDEORESIZE:
                try:
                    new_w, new_h = max(320, event.w), max(240, event.h)
                    self.screen = pygame.display.set_mode(
                        (new_w, new_h), pygame.RESIZABLE
                    )
                    if hasattr(self.renderer, "screen"):
                        self.renderer.screen = self.screen
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
                    print(f"Window resized: {new_w}x{new_h}")  # LOG
                except pygame.error as e:
                    print(f"Error resizing window: {e}")  # LOG

            # --- State-Dependent Input Handling ---

            # 1. Cleanup Confirmation Overlay (takes precedence)
            if cleanup_confirmation_active:
                # print("[InputHandler] Handling input during cleanup confirmation.") # DEBUG LOG
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("[InputHandler] Cleanup cancelled via ESC.")  # LOG
                    self.cancel_cleanup_cb()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if confirm_yes_rect.collidepoint(mouse_pos):
                        print("[InputHandler] Cleanup confirmed via YES click.")  # LOG
                        self.confirm_cleanup_cb()
                    elif confirm_no_rect.collidepoint(mouse_pos):
                        print("[InputHandler] Cleanup cancelled via NO click.")  # LOG
                        self.cancel_cleanup_cb()
                continue  # Skip other inputs

            # 2. Playing State Input
            elif app_state == "Playing":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("[InputHandler] Exiting demo mode via ESC.")  # LOG
                        self.exit_demo_mode_cb()
                    else:
                        self.handle_demo_input_cb(event)

            # 3. Main Menu State Input
            elif app_state == "MainMenu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("[InputHandler] Exiting app via ESC.")  # LOG
                        return self.exit_app_cb()
                    elif event.key == pygame.K_p:
                        print("[InputHandler] Toggle training via 'P' key.")  # LOG
                        self.toggle_training_cb()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    print(
                        f"[InputHandler] Left click detected at {mouse_pos} in MainMenu"
                    )  # LOG
                    if train_btn_rect.collidepoint(mouse_pos):
                        print("[InputHandler] Train button clicked.")  # LOG
                        self.toggle_training_cb()
                    elif cleanup_btn_rect.collidepoint(mouse_pos):
                        print("[InputHandler] Cleanup button clicked.")  # LOG
                        self.request_cleanup_cb()  # <--- Check if this callback is called
                    elif demo_btn_rect.collidepoint(mouse_pos):
                        print("[InputHandler] Demo button clicked.")  # LOG
                        self.start_demo_mode_cb()
                    else:
                        # print("[InputHandler] Clicked outside known buttons.") # LOG (Optional)
                        pass

            # 4. Error State Input
            elif app_state == "Error":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("[InputHandler] Exiting app from Error state via ESC.")  # LOG
                    return self.exit_app_cb()

        return True
