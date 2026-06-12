import pygame
import sys
import os
import yaml
import numpy as np


def view_track(track_type: str = "procedural", seed: int | None = None, track_width: float | None = None):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from env.track import Track

    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption(f"Track Viewer - {track_type}")
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    track = Track(track_type=track_type, seed=seed)
    if track_width is not None:
        track.track_width = track_width

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    seed = np.random.randint(0, 10000) if seed is None else seed + 1
                    track = Track(track_type=track_type, seed=seed)
                    if track_width is not None:
                        track.track_width = track_width
                elif event.key == pygame.K_r:
                    seed = None
                    track = Track(track_type=track_type)

        screen.fill((30, 30, 40))
        outer, inner = track.get_boundaries()
        if len(outer) > 2:
            pygame.draw.polygon(screen, (180, 180, 190), outer)
        if len(inner) > 2:
            pygame.draw.polygon(screen, (30, 30, 40), inner)
        if hasattr(track, "center_points") and len(track.center_points) > 2:
            pygame.draw.lines(screen, (80, 80, 90), True, track.center_points, 1)

        for ox, oy, radius in track.get_obstacles():
            pygame.draw.circle(screen, (200, 80, 80), (int(ox), int(oy)), int(radius))
            pygame.draw.circle(screen, (255, 100, 100), (int(ox), int(oy)), int(radius), 2)

        sx, sy, _ = track.get_start_pose()
        pygame.draw.circle(screen, (0, 255, 0), (int(sx), int(sy)), 6)

        info_lines = [
            f"Track: {track_type} | Seed: {track.seed}",
            f"Width: {track.track_width:.0f}px | Points: {len(track.outer_points) if hasattr(track, 'outer_points') else 0}",
            "SPACE: new seed  R: random  ESC: quit",
        ]
        for i, line in enumerate(info_lines):
            text = font.render(line, True, (220, 220, 220))
            screen.blit(text, (10, 10 + i * 22))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize procedural tracks")
    parser.add_argument("--type", default="procedural", choices=["procedural", "oval", "figure_8", "multi_loop"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=float, default=None)
    args = parser.parse_args()
    view_track(args.type, args.seed, args.width)


if __name__ == "__main__":
    main()
