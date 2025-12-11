# Spirograph & Gear Simulation

A Python simulation that visualizes the mathematical generation of **Epicycloids** and **Hypocycloids**. You can choose to visualize these paths using standard circles or gears

## Demos

| **Gear Mode** | **Wheel Mode** |
|:---:|:---:|
| ![Gear Simulation](./gifs/gears.gif) | ![Wheel Simulation](./gifs/wheels.gif) |

## Features

* **Path Types:**
    * **Epicycloid:** The moving shape rolls on the *outside* of the base shape.
    * **Hypocycloid:** The moving shape rolls on the *inside* of the base shape.
* **Shape Modes:**
    * **Wheels:** Simple circles visualizing the tangent points.
    * **Gears:** Procedurally generated gears with teeth calculated to match the radius ratio.
* **Tracing point choice:** Choose the point using polar coordinates relative to the center of the moving shape (in code)
* **Real-time Animation:** Visualizes the tracing point, the drawing arm, and the resulting path history.

## Requirements

This project requires **Python 3** and the following libraries:

* `numpy`
* `matplotlib`
* `Pillow` (for saving GIFs)

## Installation

1. Clone this repository or download the script.
2. Install the dependencies using pip:

```bash
pip install numpy matplotlib Pillow
