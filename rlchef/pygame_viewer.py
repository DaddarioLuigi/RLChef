from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ViewerConfig:
    cell_px: int = 96
    margin_px: int = 12
    panel_px: int = 260
    fps: int = 12


class PygameViewer:
    def __init__(self, *, size: int, k: int, ingredient_names: list[str], cfg: ViewerConfig | None = None) -> None:
        import pygame

        self.pygame = pygame
        self.size = int(size)
        self.k = int(k)
        self.ingredient_names = ingredient_names
        self.cfg = cfg or ViewerConfig()

        w = self.cfg.margin_px * 3 + self.cfg.cell_px * self.size + self.cfg.panel_px
        h = self.cfg.margin_px * 2 + self.cfg.cell_px * self.size

        pygame.init()
        pygame.display.set_caption("RL Chef")
        self.screen = pygame.display.set_mode((w, h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 16) or pygame.font.SysFont(None, 16)
        self.big = pygame.font.SysFont("Menlo", 18, bold=True) or pygame.font.SysFont(None, 18)

        self.colors = self._palette(self.k)

    def close(self) -> None:
        self.pygame.quit()

    def tick(self, fps: int | None = None) -> None:
        self.clock.tick(fps or self.cfg.fps)

    def handle_events(self) -> bool:
        for e in self.pygame.event.get():
            if e.type == self.pygame.QUIT:
                return False
        return True

    def render(
        self,
        *,
        grid: np.ndarray,
        available: np.ndarray,
        pos: tuple[int, int],
        inv: np.ndarray,
        last_reward: float | None = None,
        made: str | None = None,
        step: int | None = None,
        max_steps: int | None = None,
        budget: float | None = None,
        round_idx: int | None = None,
        rounds_total: int | None = None,
    ) -> bool:
        pg = self.pygame
        cfg = self.cfg

        self.screen.fill((18, 18, 22))

        ox = cfg.margin_px
        oy = cfg.margin_px
        s = self.size

        for i in range(s):
            for j in range(s):
                ing = int(grid[i, j])
                x = ox + j * cfg.cell_px
                y = oy + i * cfg.cell_px

                pg.draw.rect(self.screen, self.colors[ing], (x, y, cfg.cell_px, cfg.cell_px), border_radius=10)
                pg.draw.rect(self.screen, (35, 35, 45), (x, y, cfg.cell_px, cfg.cell_px), width=2, border_radius=10)

                if int(available[i, j]) == 0:
                    overlay = pg.Surface((cfg.cell_px, cfg.cell_px), pg.SRCALPHA)
                    overlay.fill((0, 0, 0, 150))
                    self.screen.blit(overlay, (x, y))

                label = self.font.render(str(ing), True, (15, 15, 18))
                self.screen.blit(label, (x + 8, y + 6))

        ax = ox + pos[1] * cfg.cell_px + cfg.cell_px // 2
        ay = oy + pos[0] * cfg.cell_px + cfg.cell_px // 2
        pg.draw.circle(self.screen, (245, 245, 248), (ax, ay), int(cfg.cell_px * 0.18))
        pg.draw.circle(self.screen, (20, 20, 25), (ax, ay), int(cfg.cell_px * 0.18), width=3)

        panel_x = ox + s * cfg.cell_px + cfg.margin_px * 2
        panel_y = oy
        pg.draw.rect(
            self.screen,
            (25, 25, 30),
            (panel_x, panel_y, cfg.panel_px, s * cfg.cell_px),
            border_radius=14,
        )
        pg.draw.rect(
            self.screen,
            (40, 40, 50),
            (panel_x, panel_y, cfg.panel_px, s * cfg.cell_px),
            width=2,
            border_radius=14,
        )

        y = panel_y + 14
        title = self.big.render("Inventory", True, (235, 235, 240))
        self.screen.blit(title, (panel_x + 14, y))
        y += 28

        for idx, name in enumerate(self.ingredient_names):
            txt = f"{idx}: {name}  x{int(inv[idx])}"
            line = self.font.render(txt, True, (210, 210, 220))
            self.screen.blit(line, (panel_x + 14, y))
            y += 20

        y += 10
        if step is not None and max_steps is not None:
            line = self.font.render(f"step: {step}/{max_steps}", True, (200, 200, 210))
            self.screen.blit(line, (panel_x + 14, y))
            y += 20

        if round_idx is not None and rounds_total is not None:
            line = self.font.render(f"round: {round_idx}/{rounds_total}", True, (200, 200, 210))
            self.screen.blit(line, (panel_x + 14, y))
            y += 20

        if budget is not None:
            line = self.font.render(f"budget: {budget:.2f}", True, (200, 200, 210))
            self.screen.blit(line, (panel_x + 14, y))
            y += 20

        if last_reward is not None:
            line = self.font.render(f"reward: {last_reward:+.3f}", True, (200, 200, 210))
            self.screen.blit(line, (panel_x + 14, y))
            y += 20

        if made is not None:
            line = self.font.render(f"made: {made}", True, (230, 230, 240))
            self.screen.blit(line, (panel_x + 14, y))

        pg.display.flip()
        return self.handle_events()

    def _palette(self, k: int) -> list[tuple[int, int, int]]:
        base = [
            (231, 76, 60),
            (241, 196, 15),
            (46, 204, 113),
            (52, 152, 219),
            (155, 89, 182),
            (230, 126, 34),
            (26, 188, 156),
            (149, 165, 166),
        ]
        if k <= len(base):
            return base[:k]
        out = []
        for i in range(k):
            out.append(base[i % len(base)])
        return out

