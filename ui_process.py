# File: ui_process.py
# Contains the run_ui_process function and necessary UI-related imports

import pygame
import sys
import time
import queue
import logging
import logging.handlers
import multiprocessing as mp
from typing import Optional, Dict, Any

try:
    from config import VisConfig, BLACK, RED, WHITE  # Import necessary constants
    from app_state import AppState
    from ui.renderer import UIRenderer
    from ui.input_handler import InputHandler
    from app_setup import initialize_pygame
except ImportError as e:
    print(f"[UI Process Import Error] {e}", file=sys.stderr)

RENDER_DATA_SENTINEL = "RENDER_DATA"
STOP_SENTINEL = "STOP"
ERROR_SENTINEL = "ERROR"
UI_QUEUE_GET_TIMEOUT = 0.01


def run_ui_process(
    stop_event: mp.Event,
    command_queue: mp.Queue,
    render_data_queue: mp.Queue,
    log_queue: Optional[mp.Queue] = None,
):
    if log_queue:
        qh = logging.handlers.QueueHandler(log_queue)
        root = logging.getLogger()
        if not any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers):
            root.addHandler(qh)
            root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("UI Process starting...")

    vis_config = VisConfig()
    screen: Optional[pygame.Surface] = None
    clock: Optional[pygame.time.Clock] = None
    renderer: Optional[UIRenderer] = None
    input_handler: Optional[InputHandler] = None
    last_render_data: Dict[str, Any] = {}
    running = True

    try:
        screen, clock = initialize_pygame(vis_config)
        renderer = UIRenderer(screen, vis_config)
        input_handler = InputHandler(screen, renderer, command_queue, stop_event)
        renderer.set_input_handler(input_handler)
        logger.info("Pygame and UI components initialized.")
    except Exception as init_err:
        logger.critical(f"UI Initialization failed: {init_err}", exc_info=True)
        stop_event.set()
        running = False

    while running and not stop_event.is_set():
        if not clock or not screen or not renderer or not input_handler:
            logger.error("Critical UI components not initialized. Exiting UI loop.")
            stop_event.set()
            break

        # --- Handle Input ---
        try:
            input_handler.update_state(
                last_render_data.get("app_state", AppState.INITIALIZING.value),
                last_render_data.get("cleanup_confirmation_active", False),
                last_render_data.get("is_process_running", False),  # Pass worker status
            )
            running = input_handler.handle_input()  # This might set stop_event
            if not running:
                logger.info("Input handler requested exit.")
                break
        except Exception as input_err:
            logger.error(f"Error handling input: {input_err}", exc_info=True)
            stop_event.set()
            running = False
            break

        # --- Get Render Data ---
        try:
            item = render_data_queue.get(timeout=UI_QUEUE_GET_TIMEOUT)
            if isinstance(item, dict) and RENDER_DATA_SENTINEL in item:
                last_render_data = item[RENDER_DATA_SENTINEL]
            elif item == STOP_SENTINEL:
                logger.info("Received STOP sentinel from logic process.")
                running = False
                break
            elif isinstance(item, dict) and ERROR_SENTINEL in item:
                logger.error(f"Received ERROR sentinel: {item[ERROR_SENTINEL]}")
                last_render_data = last_render_data.copy()
                last_render_data["app_state"] = AppState.ERROR.value
                last_render_data["status"] = item[ERROR_SENTINEL]
        except queue.Empty:
            pass  # No new data
        except (EOFError, BrokenPipeError):
            logger.warning("Render data queue connection lost.")
            running = False
            stop_event.set()
        except Exception as queue_err:
            logger.error(f"Error getting render data: {queue_err}", exc_info=True)

        # --- Render Frame ---
        if last_render_data:
            try:
                renderer.render_all(**last_render_data)
            except Exception as render_err:
                logger.error(f"Error rendering frame: {render_err}", exc_info=True)
                try:  # Fallback render
                    screen.fill(BLACK)
                    font = pygame.font.Font(None, 30)
                    err_surf = font.render("Error during rendering!", True, RED)
                    screen.blit(
                        err_surf, err_surf.get_rect(center=screen.get_rect().center)
                    )
                    pygame.display.flip()
                except Exception:
                    pass
        else:  # Render waiting screen
            try:
                screen.fill(BLACK)
                font = pygame.font.Font(None, 30)
                wait_surf = font.render("Waiting for data...", True, WHITE)
                screen.blit(
                    wait_surf, wait_surf.get_rect(center=screen.get_rect().center)
                )
                pygame.display.flip()
            except Exception:
                pass

        clock.tick(vis_config.FPS if vis_config.FPS > 0 else 60)

    logger.info("UI Process shutting down...")
    pygame.quit()
    logger.info("UI Process finished.")
