"""A mini strategy game implementation with various AI players."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from PIL import Image, ImageDraw  # type: ignore

# Constants
ROOT_DIR = Path()
SPRITE_DIR = ROOT_DIR / "data"
STATE_DIM = 474 + 14  # Base features + statistical features
N_ACTIONS = 9
FIELD_SIZE = 48
SCALE = 6


@dataclass
class UnitType:
    """Represents a type of unit in the game with its attributes."""

    type: str
    maxhp: float
    cooldown: int
    armour_type: str  # 'inf', 'cavalry', 'heavy_inf'
    att_type: str  # 'miner', 'pike', 'arrow', 'sword', 'heal'
    r: float  # Unit radius
    dv: float  # Movement speed
    att_range: float  # Attack range
    price: int  # Cost to create unit


def encode_simple_colors(image: np.ndarray) -> np.ndarray:
    """Encode an image into a feature vector by measuring brightness in different grid patterns."""
    features = []
    shape = image.shape

    for color_channel in range(3):
        # Sample in 10x10 grid
        for grid_size in [10, 7, 3]:
            cell_h = shape[0] // grid_size
            cell_w = shape[1] // grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    cell = image[
                        i * cell_h : (i + 1) * cell_h,
                        j * cell_w : (j + 1) * cell_w,
                        color_channel,
                    ]
                    features.append(np.nanmean(cell))

    return np.array(features)


def white_to_transparency(img: Image.Image) -> Image.Image:
    """Convert white pixels in an image to transparent."""
    rgba = np.asarray(img.convert("RGBA")).copy()
    rgba[:, :, 3] = (255 * (rgba[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
    return Image.fromarray(rgba)


@dataclass
class Unit:
    """Represents a unit in the game."""

    id: int
    type: str
    team: int
    x: float
    y: float
    hp: float
    maxhp: float
    cooldown: int
    t: int = 0
    target: Optional[int] = None
    armour_type: str = "inf"
    att_type: str = "miner"
    r: float = 0.1
    dv: float = 0.9
    att_range: float = 1
    trg_x: float = -1
    trg_y: float = -1


class Player(ABC):
    """Abstract base class for all player types."""

    @abstractmethod
    def act(self, img: Image.Image, game: StrategyGame, team: int) -> List[float]:
        """Generate action vector based on game state."""
        pass


class WorkerRushPlayer(Player):
    """Player that focuses on creating workers early game."""

    def __init__(self):
        self.turn = 0

    def act(self, img: Image.Image, game: StrategyGame, team: int) -> List[float]:
        x = np.random.rand()
        y = np.random.rand()

        if self.turn <= 4:
            # Build workers early
            action = [1, 0, 0, 0, 0, 0, x, y]
        else:
            # Stop building
            action = [0, 0, 0, 0, 0, 0, 0, 0]

        self.turn += 1
        return action


class BalancedPlayer(Player):
    """Player that tries to maintain a balanced army composition."""

    def act(self, img: Image.Image, game: StrategyGame, team: int) -> List[float]:
        x = np.random.rand()
        y = np.random.rand()

        # Count unit types
        workers = 0
        military = 0
        enemy_military = 0

        for unit in game.units:
            if unit.hp <= 0:
                continue

            if unit.team == team:
                if unit.type == "worker":
                    workers += 1
                elif unit.type != "medic":
                    military += 1
            else:
                if unit.type not in ["worker", "medic"]:
                    enemy_military += 1

        # Default - no action
        action = [0, 0, 0, 0, 0, 0, x, y]

        # Build workers if needed
        if workers == 0:
            action = [1, 0, 0, 0, 0, 0, x, y]
        # Build military if outnumbered
        elif military < enemy_military:
            action = [0, 0, 1, 0, 0, 0, x, y]  # Swordsman
        # Build knight if no military
        elif military == 0:
            action = [0, 0, 0, 0, 0, 1, x, y]  # Knight

        return action


class AdaptivePlayer(Player):
    """Player that adapts unit composition to counter enemy units."""

    def act(self, img: Image.Image, game: StrategyGame, team: int) -> List[float]:
        x = np.random.rand()
        y = np.random.rand()

        # Count enemy unit types
        workers = 0
        spearmen = 0
        archers = 0
        cavalry = 0
        military = 0
        enemy_military = 0

        for unit in game.units:
            if unit.hp <= 0:
                continue

            if unit.team == team:
                if unit.type == "worker":
                    workers += 1
                elif unit.type != "medic":
                    military += 1
            else:
                if unit.type == "pikeman":
                    spearmen += 1
                elif unit.type == "archer":
                    archers += 1
                elif unit.type == "knight":
                    cavalry += 2
                if unit.type not in ["worker", "medic"]:
                    enemy_military += 1

        # Default - no action
        action = [0, 0, 0, 0, 0, 0, x, y]

        # Build workers if needed
        if workers == 0:
            action = [1, 0, 0, 0, 0, 0, x, y]

        # Counter enemy composition
        elif military < enemy_military:
            enemy_comp = [spearmen, archers, cavalry]
            strongest = np.argmax(enemy_comp)

            if strongest == 0:  # Counter spearmen
                if spearmen == cavalry:
                    action = [0, 0, 1, 0, 0, 0, x, y]  # Swordsman
                else:
                    action = [0, 0, 0, 1, 0, 0, x, y]  # Archer
            elif strongest == 1:  # Counter archers
                action = [0, 0, 0, 0, 0, 1, x, y]  # Knight
            else:  # Counter cavalry
                action = [0, 1, 0, 0, 0, 0, x, y]  # Pikeman

        return action


class ArcherPlayer(Player):
    """Player that focuses on archer units."""

    def act(self, img: Image.Image, game: StrategyGame, team: int) -> List[float]:
        x = np.random.rand()
        y = np.random.rand()

        # Count workers
        workers = sum(
            1
            for unit in game.units
            if unit.hp > 0 and unit.team == team and unit.type == "worker"
        )

        # Build workers if needed
        if workers == 0:
            return [1, 0, 0, 0, 0, 0, x, y]
        # Otherwise build archers
        return [0, 0, 0, 1, 0, 0, x, y]


class ArmyPlayer(Player):
    """Player that builds a diverse army composition."""

    def __init__(self):
        self.turn = 0

    def act(self, img: Image.Image, game: StrategyGame, team: int) -> List[float]:
        x = np.random.rand()
        y = np.random.rand()

        # Build order
        if self.turn == 1:
            action = [1, 0, 0, 0, 0, 0, x, y]  # Worker
        elif self.turn == 2:
            action = [0, 0, 1, 0, 0, 0, x, y]  # Swordsman
        elif self.turn == 3:
            action = [0, 0, 0, 0, 1, 0, x, y]  # Medic
        elif self.turn == 4:
            action = [0, 0, 0, 1, 0, 0, x, y]  # Archer
        else:
            # Count units
            workers = 0
            military = 0
            enemy_military = 0

            for unit in game.units:
                if unit.hp <= 0:
                    continue

                if unit.team == team:
                    if unit.type == "worker":
                        workers += 1
                    elif unit.type != "medic":
                        military += 1
                else:
                    if unit.type not in ["worker", "medic"]:
                        enemy_military += 1

            # Build workers if needed
            if workers == 0:
                action = [1, 0, 0, 0, 0, 0, x, y]
            # Build archers if outnumbered
            elif enemy_military > military:
                action = [0, 0, 0, 1, 0, 0, x, y]
            else:
                action = [0, 0, 0, 0, 0, 0, x, y]

        self.turn += 1
        return action


class StrategyGame:
    """Main game class implementing the strategy game mechanics."""

    def __init__(self):
        """Initialize game state."""
        # Initialize game field with random resource distribution
        self.field = np.round((np.random.rand(FIELD_SIZE, FIELD_SIZE) ** 150) * 12)
        self.money = [50, 50]  # Starting money for both players
        self.units: List[Unit] = []
        self.sprite_base: Dict[str, Image.Image] = {}

        # Initialize unit database
        self.unit_db = {
            "worker": UnitType("worker", 1, 2, "inf", "miner", 0.1, 0.9, 1, 7),
            "pikeman": UnitType("pikeman", 2, 5, "inf", "pike", 0.1, 0.9, 4, 5),
            "swordsman": UnitType(
                "swordsman", 2, 4, "heavy_inf", "sword", 0.1, 0.7, 1, 7
            ),
            "archer": UnitType("archer", 1.5, 14, "inf", "arrow", 0.1, 0.9, 15, 7),
            "medic": UnitType("medic", 1.6, 12, "inf", "heal", 0.1, 1.1, 1, 7),
            "knight": UnitType("knight", 4, 2, "cavalry", "sword", 0.3, 1.9, 1.3, 16),
        }

    def get_sprite(self, sprite_name: str) -> Image.Image:
        """Load and cache sprite image."""
        if sprite_name not in self.sprite_base:
            sprite_path = SPRITE_DIR / f"{sprite_name}.bmp"
            sprite = Image.open(sprite_path)
            self.sprite_base[sprite_name] = white_to_transparency(sprite)
        return self.sprite_base[sprite_name]

    def add_unit(self, unit_type: str, coords: Tuple[float, float], team: int) -> None:
        """Add a new unit to the game."""
        unit_template = self.unit_db[unit_type]
        unit = Unit(
            id=len(self.units),
            type=unit_type,
            team=team,
            x=coords[0],
            y=coords[1],
            hp=unit_template.maxhp,
            maxhp=unit_template.maxhp,
            cooldown=unit_template.cooldown,
            armour_type=unit_template.armour_type,
            att_type=unit_template.att_type,
            r=unit_template.r,
            dv=unit_template.dv,
            att_range=unit_template.att_range,
        )
        self.units.append(unit)

    def step_env(self, draw: bool = False) -> Image.Image:
        """Advance game state by one step and render the game."""
        # Create base image
        im = Image.new("RGB", (FIELD_SIZE * SCALE, FIELD_SIZE * SCALE), (0, 64, 0))
        dr = ImageDraw.Draw(im)

        # Draw resources
        self._draw_resources(dr, im)

        # Update and draw units
        self._update_units(dr, im)

        # Draw UI elements
        self._draw_ui(dr)

        if draw:
            im = im.resize((900, 900))
            if "video" in globals():
                globals()["video"].append(im)

        return im

    def _draw_resources(self, dr: Any, im: Image.Image) -> None:
        """Draw resource fields on the game map."""
        for x in range(FIELD_SIZE):
            for y in range(FIELD_SIZE):
                if self.field[x, y] > 0:
                    # Choose color based on resource amount
                    if self.field[x, y] < 2:
                        color = "green"
                    elif self.field[x, y] < 4:
                        color = "lime"
                    elif self.field[x, y] < 7:
                        color = "gold"
                    else:
                        color = "yellow"

                    # Draw resource square
                    shape = [
                        ((x - 0.5) * SCALE, (y - 0.5) * SCALE),
                        ((x + 0.5) * SCALE, (y + 0.5) * SCALE),
                    ]
                    dr.rectangle(shape, fill=color)

                    # Draw resource sprite
                    sprite = self.get_sprite("gold")
                    sprite = sprite.resize((SCALE, SCALE))
                    im.paste(
                        sprite,
                        (int((x - 0.5) * SCALE), int((y - 0.5) * SCALE)),
                        sprite.split()[-1],
                    )

    def _update_units(self, dr: Any, im: Image.Image) -> None:
        """Update unit states and draw them."""
        for unit in self.units:
            if unit.hp <= 0:
                continue

            # Draw unit
            shape = [
                ((unit.x - 0.5) * SCALE, (unit.y - 0.5) * SCALE),
                ((unit.x + 0.5) * SCALE, unit.y * SCALE),
            ]
            team_color = "red" if unit.team == 0 else "blue"
            dr.rectangle(shape, fill=team_color)

            # Draw unit sprite
            sprite = self.get_sprite(unit.type)
            if unit.team == 1:
                # Flip image horizontally using numpy array manipulation
                sprite_array = np.array(sprite)
                sprite_array = sprite_array[:, ::-1]
                sprite = Image.fromarray(sprite_array)
            im.paste(
                sprite, (int(unit.x * SCALE), int(unit.y * SCALE)), sprite.split()[-1]
            )

            # Update unit state
            self._update_unit_state(unit)

    def _update_unit_state(self, unit: Unit) -> None:
        """Update a single unit's state including targeting and movement."""
        unit.t += 1

        # Update target (30% chance or if no target)
        if np.random.rand() < 0.3 or not hasattr(unit, "trg"):
            self._update_unit_target(unit)

        # Move or attack
        if unit.target is not None:
            self._handle_unit_action(unit)

    def _update_unit_target(self, unit: Unit) -> None:
        """Update unit's target."""
        if unit.att_type != "miner":
            self._update_combat_target(unit)
        else:
            self._update_mining_target(unit)

    def _update_combat_target(self, unit: Unit) -> None:
        """Find closest valid combat target for unit."""
        unit.target = None
        closest_dist = float("inf")

        for i, target in enumerate(self.units):
            if target.hp <= 0:
                continue

            # Healers target damaged friendly units, others target enemies
            valid_target = (
                unit.att_type == "heal"
                and target.team == unit.team
                and target.hp < target.maxhp
            ) or (unit.att_type != "heal" and target.team != unit.team)

            if valid_target:
                dist = abs(target.x - unit.x) + abs(target.y - unit.y)
                if dist < closest_dist:
                    closest_dist = dist
                    unit.target = i
                    unit.trg_x = target.x
                    unit.trg_y = target.y

    def _update_mining_target(self, unit: Unit) -> None:
        """Find closest resource for worker to mine."""
        for radius in range(FIELD_SIZE):
            x_min = max(0, int(unit.x - radius))
            x_max = min(FIELD_SIZE, int(unit.x + radius))
            y_min = max(0, int(unit.y - radius))
            y_max = min(FIELD_SIZE, int(unit.y + radius))

            # Check perimeter
            for x in range(x_min, x_max):
                for y in [y_min, y_max - 1]:
                    if y >= 0 and y < FIELD_SIZE and self.field[x, y] > 0:
                        unit.target = -1  # Special value for mining
                        unit.trg_x = x
                        unit.trg_y = y
                        return

            for y in range(y_min, y_max):
                for x in [x_min, x_max - 1]:
                    if x >= 0 and x < FIELD_SIZE and self.field[x, y] > 0:
                        unit.target = -1
                        unit.trg_x = x
                        unit.trg_y = y
                        return

    def _handle_unit_action(self, unit: Unit) -> None:
        """Handle unit movement and combat."""
        # Calculate movement direction
        dx = unit.trg_x - unit.x
        dy = unit.trg_y - unit.y
        dist = (dx**2 + dy**2) ** 0.5

        # If in range, handle combat
        if dist <= unit.att_range:
            self._handle_unit_combat(unit)
        # Otherwise move
        else:
            # Normalize direction
            if dist > 0:
                dx /= dist
                dy /= dist

            # Calculate new position
            new_x = unit.x + dx * unit.dv
            new_y = unit.y + dy * unit.dv

            # Keep within bounds
            new_x = max(0, min(FIELD_SIZE - 1, new_x))
            new_y = max(0, min(FIELD_SIZE - 1, new_y))

            # Check for collisions
            if not self._check_collision(unit, new_x, new_y):
                unit.x = new_x
                unit.y = new_y

    def _check_collision(self, unit: Unit, new_x: float, new_y: float) -> bool:
        """Check if moving to new position would result in collision."""
        for other in self.units:
            if other.id == unit.id or other.hp <= 0:
                continue

            dx = new_x - other.x
            dy = new_y - other.y
            dist = (dx**2 + dy**2) ** 0.5

            if dist < (unit.r + other.r):
                return True

        return False

    def _handle_unit_combat(self, unit: Unit) -> None:
        """Handle unit combat actions."""
        if unit.t % unit.cooldown != 0:
            return

        # Handle mining
        if unit.target == -1:
            x, y = int(unit.trg_x), int(unit.trg_y)
            if self.field[x, y] > 0:
                self.field[x, y] -= 1
                self.money[unit.team] += 1
            return

        # Handle combat
        target = self.units[cast(int, unit.target)]
        if target.hp <= 0:
            return

        # Calculate damage
        damage = self._calculate_damage(unit.att_type, target.armour_type)

        # Apply damage or healing
        if unit.att_type == "heal":
            target.hp = min(target.hp + damage, target.maxhp)
        else:
            target.hp -= damage

    def _calculate_damage(self, att_type: str, armour_type: str) -> float:
        """Calculate damage based on attack and armor types."""
        base_damage = {
            "miner": 0.1,
            "pike": 0.4,
            "arrow": 0.3,
            "sword": 0.35,
            "heal": 0.2,
        }[att_type]

        # Apply armor type modifiers
        modifiers = {
            ("pike", "cavalry"): 2.0,
            ("arrow", "inf"): 1.5,
            ("sword", "heavy_inf"): 0.7,
        }

        modifier = modifiers.get((att_type, armour_type), 1.0)
        return base_damage * modifier

    def _draw_ui(self, dr: Any) -> None:
        """Draw UI elements."""
        # Draw money counters
        dr.text((10, 10), f"Red: ${self.money[0]}", fill="red")
        dr.text((10, 30), f"Blue: ${self.money[1]}", fill="blue")

    def action(self, team: int, action: List[float]) -> None:
        """Process player action."""
        # Extract action components
        unit_type_probs = action[:6]
        x, y = action[6:8]

        # Determine unit type to create
        unit_types = ["worker", "pikeman", "swordsman", "archer", "medic", "knight"]
        unit_type = unit_types[np.argmax(unit_type_probs)]

        # Check if we can afford the unit
        if self.money[team] >= self.unit_db[unit_type].price:
            # Convert normalized coordinates to game coordinates
            game_x = x * FIELD_SIZE
            game_y = y * FIELD_SIZE

            # Add unit and deduct cost
            self.add_unit(unit_type, (game_x, game_y), team)
            self.money[team] -= self.unit_db[unit_type].price
