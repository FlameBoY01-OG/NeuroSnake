# visualization/activations.py
"""
Neural Network Activation Visualizer for Explainable AI.

This module handles the animated, live visualization of:
- Node activations (pulsing/glowing based on magnitude)
- Edge weights (thickness/color based on contribution)
- Decision flow (highlighting the path to selected action)
- Dueling architecture separation (value vs advantage streams)

Color scheme: Catppuccin Mocha theme
"""
import numpy as np
import pygame
from typing import Dict, List, Tuple, Optional

# Catppuccin Mocha color palette
COLORS = {
    'base': (30, 30, 46),
    'mantle': (24, 24, 37),
    'crust': (17, 17, 27),
    'text': (205, 214, 244),
    'subtext0': (166, 173, 200),
    'overlay0': (108, 112, 134),
    'surface0': (49, 50, 68),
    'red': (243, 139, 168),
    'green': (166, 227, 161),
    'blue': (137, 180, 250),
    'yellow': (249, 226, 175),
    'teal': (148, 226, 213),
    'lavender': (180, 190, 254),
    'mauve': (203, 166, 247),
    'peach': (250, 179, 135),
    'sapphire': (116, 199, 236),
}

class ActivationVisualizer:
    """
    Handles live animation of neural network activations for explainable AI.
    """
    def __init__(self):
        self.pulse_phase = 0.0  # Animation phase for pulsing effect
        self.pulse_speed = 0.1
        self.edge_alpha_decay = 0.85  # Smooth edge highlighting decay
        self.node_history = {}  # Track activation history for smooth transitions
        
    def update_animation(self, dt: float = 1.0):
        """Update animation state. Call once per frame."""
        self.pulse_phase = (self.pulse_phase + self.pulse_speed * dt) % (2 * np.pi)
    
    def get_node_color_and_size(self, activation: float, is_selected: bool = False) -> Tuple[Tuple[int, int, int], float]:
        """
        Compute node color and size based on activation magnitude.
        
        Args:
            activation: Node activation value (can be negative)
            is_selected: Whether this is the selected action node
            
        Returns:
            (color_rgb, size_multiplier)
        """
        abs_act = abs(activation)
        
        # Base color selection
        if is_selected:
            # Selected action glows bright
            base_color = COLORS['green']
            pulse_intensity = 0.3 + 0.3 * np.sin(self.pulse_phase)
        elif activation > 0:
            # Positive activations: blue gradient
            base_color = COLORS['blue']
            pulse_intensity = 0.1 * np.sin(self.pulse_phase)
        elif activation < 0:
            # Negative activations: red gradient
            base_color = COLORS['red']
            pulse_intensity = 0.1 * np.sin(self.pulse_phase)
        else:
            # Inactive: gray
            base_color = COLORS['overlay0']
            pulse_intensity = 0
        
        # Compute intensity-based brightness
        intensity = min(1.0, abs_act / 3.0)  # Normalize to [0, 1]
        brightness_factor = 0.3 + 0.7 * intensity + pulse_intensity
        brightness_factor = np.clip(brightness_factor, 0.2, 1.3)
        
        # Apply brightness
        color = tuple(int(c * brightness_factor) for c in base_color)
        color = tuple(np.clip(c, 0, 255) for c in color)
        
        # Size scales with activation magnitude
        size_multiplier = 0.7 + 0.6 * intensity
        if is_selected:
            size_multiplier *= 1.4  # Selected action is bigger
        
        return color, size_multiplier
    
    def get_edge_color_and_alpha(self, weight: float, src_activation: float, dst_activation: float) -> Tuple[Tuple[int, int, int], int]:
        """
        Compute edge color and transparency based on weight and activation flow.
        
        Args:
            weight: Connection weight
            src_activation: Source node activation
            dst_activation: Destination node activation
            
        Returns:
            (color_rgb, alpha_0_255)
        """
        # Edge contribution: weight * src_activation
        contribution = abs(weight * src_activation)
        
        # Normalize contribution to [0, 1] range
        contribution_normalized = min(1.0, contribution / 2.0)
        
        # Color based on weight sign
        if weight > 0:
            base_color = COLORS['teal']  # Positive weights: teal
        else:
            base_color = COLORS['peach']  # Negative weights: peach
        
        # Alpha based on contribution magnitude
        alpha = int(30 + 200 * contribution_normalized)
        alpha = np.clip(alpha, 10, 255)
        
        return base_color, alpha
    
    def draw_neuron_layer(
        self,
        surface: pygame.Surface,
        activations: np.ndarray,
        positions: List[Tuple[int, int]],
        base_radius: float = 8,
        labels: Optional[List[str]] = None,
        selected_indices: Optional[List[int]] = None
    ):
        """
        Draw a layer of neurons with animated activations.
        
        Args:
            surface: Pygame surface to draw on
            activations: Array of activation values
            positions: List of (x, y) coordinates for each neuron
            base_radius: Base neuron circle radius
            labels: Optional text labels for neurons
            selected_indices: Indices of neurons to highlight as selected
        """
        selected_set = set(selected_indices or [])
        
        for i, (act, pos) in enumerate(zip(activations, positions)):
            is_selected = i in selected_set
            color, size_mult = self.get_node_color_and_size(act, is_selected)
            radius = int(base_radius * size_mult)
            
            # Draw glow effect for high activations
            if abs(act) > 1.0 or is_selected:
                glow_radius = radius + 4
                glow_color = tuple(int(c * 0.5) for c in color)
                pygame.draw.circle(surface, glow_color, pos, glow_radius)
            
            # Draw main neuron circle
            pygame.draw.circle(surface, color, pos, radius)
            pygame.draw.circle(surface, COLORS['text'], pos, radius, 1)  # Outline
            
            # Draw label if provided
            if labels and i < len(labels):
                font = pygame.font.Font(None, 16)
                text = font.render(labels[i], True, COLORS['subtext0'])
                text_rect = text.get_rect(center=(pos[0], pos[1] + radius + 10))
                surface.blit(text, text_rect)
    
    def draw_connections(
        self,
        surface: pygame.Surface,
        weights: np.ndarray,
        src_activations: np.ndarray,
        dst_activations: np.ndarray,
        src_positions: List[Tuple[int, int]],
        dst_positions: List[Tuple[int, int]],
        selected_dst_idx: Optional[int] = None,
        highlight_path: bool = True
    ):
        """
        Draw animated connections between layers with weight-based styling.
        
        Args:
            surface: Pygame surface to draw on
            weights: Weight matrix (dst_neurons x src_neurons)
            src_activations: Source layer activations
            dst_activations: Destination layer activations
            src_positions: Source neuron positions
            dst_positions: Destination neuron positions
            selected_dst_idx: Index of selected destination neuron (highlights relevant edges)
            highlight_path: Whether to highlight the decision path
        """
        # Create a temporary surface for alpha blending
        temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        
        for dst_idx in range(weights.shape[0]):
            for src_idx in range(weights.shape[1]):
                weight = weights[dst_idx, src_idx]
                src_act = src_activations[src_idx]
                dst_act = dst_activations[dst_idx]
                
                # Skip very weak connections (optimization)
                if abs(weight * src_act) < 0.01:
                    continue
                
                # Get edge styling
                color, alpha = self.get_edge_color_and_alpha(weight, src_act, dst_act)
                
                # Highlight path to selected output
                if highlight_path and selected_dst_idx is not None and dst_idx == selected_dst_idx:
                    # Brighten edges leading to selected action
                    contribution = abs(weight * src_act)
                    if contribution > 0.1:
                        alpha = min(255, int(alpha * 1.5))
                        line_width = 2
                    else:
                        line_width = 1
                else:
                    line_width = 1
                
                # Draw edge
                start_pos = src_positions[src_idx]
                end_pos = dst_positions[dst_idx]
                edge_color = (*color, alpha)
                pygame.draw.line(temp_surface, edge_color, start_pos, end_pos, line_width)
        
        # Blit temp surface onto main surface
        surface.blit(temp_surface, (0, 0))


def compute_neuron_positions(
    layer_sizes: List[int],
    area: Tuple[int, int, int, int],  # (x, y, width, height)
    vertical_spacing: float = 0.8
) -> List[List[Tuple[int, int]]]:
    """
    Compute positions for neurons in a multi-layer network visualization.
    
    Args:
        layer_sizes: Number of neurons in each layer
        area: (x, y, width, height) bounding box for the network
        vertical_spacing: Fraction of height to use for vertical spacing
        
    Returns:
        List of position lists, one per layer
    """
    x, y, width, height = area
    num_layers = len(layer_sizes)
    
    # Horizontal spacing between layers
    if num_layers > 1:
        layer_spacing = width / (num_layers - 1)
    else:
        layer_spacing = 0
    
    all_positions = []
    
    for layer_idx, layer_size in enumerate(layer_sizes):
        # X position for this layer
        layer_x = x + layer_idx * layer_spacing
        
        # Vertical positions for neurons in this layer
        usable_height = height * vertical_spacing
        if layer_size > 1:
            neuron_spacing = usable_height / (layer_size - 1)
        else:
            neuron_spacing = 0
        
        start_y = y + (height - usable_height) / 2
        
        layer_positions = []
        for neuron_idx in range(layer_size):
            neuron_y = start_y + neuron_idx * neuron_spacing
            layer_positions.append((int(layer_x), int(neuron_y)))
        
        all_positions.append(layer_positions)
    
    return all_positions
