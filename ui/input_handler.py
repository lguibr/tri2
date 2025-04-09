# File: ui/input_handler.py
import pygame
import multiprocessing as mp
from typing import Tuple, Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .renderer import UIRenderer
    from app_state import AppState

COMMAND_SENTINEL = "COMMAND"
STOP_SENTINEL = "STOP"
PAYLOAD_KEY = "payload"  # Use this key for data


class InputHandler:
    """Handles Pygame events and sends commands to the Logic process via a queue."""

    def __init__(
        self,
        screen: pygame.Surface,
        renderer: "UIRenderer",
        command_queue: mp.Queue,
        stop_event: mp.Event,
    ):
        self.screen = screen
        self.renderer = renderer
        self.command_queue = command_queue
        self.stop_event = stop_event
        self.app_state_str = "Initializing"
        self.cleanup_confirmation_active = False
        self.is_process_running_cache = False  # Cache running state

        self.run_stop_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.cleanup_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.demo_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.debug_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_yes_rect = pygame.Rect(0, 0, 0, 0)
        self.confirm_no_rect = pygame.Rect(0, 0, 0, 0)
        self.shape_preview_rects: Dict[int, pygame.Rect] = {}

        self._button_renderer = getattr(
            renderer.left_panel, "button_status_renderer", None
        )
        self._update_button_rects()

    def _update_ui_screen_references(self, new_screen: pygame.Surface):
        self.screen = new_screen
        components_to_update = [
            self.renderer,
            getattr(self.renderer, "left_panel", None),
            getattr(self.renderer, "game_area", None),
            getattr(self.renderer, "overlays", None),
            getattr(self.renderer, "demo_renderer", None),
        ]
        for component in components_to_update:
            if component and hasattr(component, "screen"):
                component.screen = new_screen

    def _update_button_rects(self):
        button_height = 40
        button_y_pos = 10
        run_stop_button_width = 150
        cleanup_button_width = 160
        demo_button_width = 120
        debug_button_width = 120
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

    def update_state(
        self,
        app_state_str: str,
        cleanup_confirmation_active: bool,
        is_process_running: bool = False,
    ):
        self.app_state_str = app_state_str
        self.cleanup_confirmation_active = cleanup_confirmation_active
        self.is_process_running_cache = is_process_running  # Update cache
        if (
            self.app_state_str == "Playing"
            and self.renderer
            and self.renderer.demo_renderer
        ):
            self.shape_preview_rects = (
                self.renderer.demo_renderer.get_shape_preview_rects()
            )
        else:
            self.shape_preview_rects.clear()

    def _send_command(self, command: str, payload: Optional[Dict] = None):
        cmd_dict = {COMMAND_SENTINEL: command}
        if payload:
            cmd_dict[PAYLOAD_KEY] = payload
        try:
            self.command_queue.put(cmd_dict)
        except Exception as e:
            print(f"Error sending command '{command}' to queue: {e}")

    def handle_input(self) -> bool:
        from app_state import AppState

        try:
            mouse_pos = pygame.mouse.get_pos()
        except pygame.error:
            mouse_pos = (0, 0)

        self._update_button_rects()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_event.set()
                self._send_command(STOP_SENTINEL)
                return False

            if event.type == pygame.VIDEORESIZE:
                try:
                    new_w, new_h = max(320, event.w), max(240, event.h)
                    new_screen = pygame.display.set_mode(
                        (new_w, new_h), pygame.RESIZABLE
                    )
                    self._update_ui_screen_references(new_screen)
                    self._update_button_rects()
                    if hasattr(self.renderer, "force_redraw"):
                        self.renderer.force_redraw()
                    print(f"Window resized: {new_w}x{new_h}")
                except pygame.error as e:
                    print(f"Error resizing window: {e}")
                continue

            if self.cleanup_confirmation_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._send_command("cancel_cleanup")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.confirm_yes_rect.collidepoint(mouse_pos):
                        self._send_command("confirm_cleanup")
                    elif self.confirm_no_rect.collidepoint(mouse_pos):
                        self._send_command("cancel_cleanup")
                continue

            current_app_state = (
                AppState(self.app_state_str)
                if self.app_state_str in AppState._value2member_map_
                else AppState.UNKNOWN
            )

            if current_app_state == AppState.PLAYING:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._send_command("exit_demo_mode")
                elif event.type == pygame.MOUSEMOTION:
                    grid_coords = None  # TODO: Implement UI-side mapping
                    self._send_command(
                        "demo_mouse_motion", payload={"pos": grid_coords}
                    )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked_preview = None
                    for idx, rect in self.shape_preview_rects.items():
                        if rect.collidepoint(event.pos):
                            clicked_preview = idx
                            break
                    if clicked_preview is not None:
                        self._send_command(
                            "demo_mouse_button_down",
                            payload={"type": "preview", "index": clicked_preview},
                        )
                    else:
                        grid_coords = None  # TODO: Implement UI-side mapping
                        if grid_coords:
                            self._send_command(
                                "demo_mouse_button_down",
                                payload={"type": "grid", "grid_coords": grid_coords},
                            )
                        else:
                            self._send_command(
                                "demo_mouse_button_down", payload={"type": "outside"}
                            )

            elif current_app_state == AppState.DEBUG:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._send_command("exit_debug_mode")
                    elif event.key == pygame.K_r:
                        self._send_command("debug_input", payload={"type": "reset"})
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    grid_coords = None  # TODO: Implement UI-side mapping
                    if grid_coords:
                        self._send_command(
                            "debug_input",
                            payload={
                                "type": "toggle_triangle",
                                "grid_coords": grid_coords,
                            },
                        )

            elif current_app_state == AppState.MAIN_MENU:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.stop_event.set()
                    self._send_command(STOP_SENTINEL)
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    is_running = (
                        self.is_process_running_cache
                    )  # Use cached value from render data

                    if self.run_stop_btn_rect.collidepoint(mouse_pos):
                        cmd = "stop_run" if is_running else "start_run"
                        if is_running:
                            self.stop_event.set()  # Set stop event immediately on UI action
                        self._send_command(cmd)
                    elif not is_running:
                        if self.cleanup_btn_rect.collidepoint(mouse_pos):
                            self._send_command("request_cleanup")
                        elif self.demo_btn_rect.collidepoint(mouse_pos):
                            self._send_command("start_demo_mode")
                        elif self.debug_btn_rect.collidepoint(mouse_pos):
                            self._send_command("start_debug_mode")

            elif current_app_state == AppState.ERROR:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.stop_event.set()
                    self._send_command(STOP_SENTINEL)
                    return False

        return True
