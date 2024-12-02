# Forest Fires Industrial Mathematics
## Overview
This project is a an agent based model to simulate the spread of fire across a grid under various environmental conditions, including wind, rain, and varying vegetation densities.

## Features
### Environmental Factors.
  - Wind effects with directional control.
  - Rainfall to simulate suppression.
  - Varying vegetation densities affecting burn and growth rates.

  **NOTE**: We generate rain and density using Perlin noise and we use the following library [https://github.com/salaxieb/perlin_noise/tree/master](Salaxieb's Perlin noise Generator).

### Custom Visualisation Settings.
  - High, medium, or low-resolution grid options.
  - Personal image integration for custom grid backgrounds.

### Planes!
- Specify parameters of your plane in `settings.json` and watch it put out a fire!

## File Structure

```text
app/
├── perlin_noise/
│   ├── perlin_noise.py
│   ├── rand_vec.py
│   └── tools.py
│
├── plane_behaviour/
│   ├── plane.py
│   └── clustering.py
│
├── animate.py
├── * simulate.py
├── * image_processor.py
└── utils.py

results/
├── fire_sim.gif
├── processed.npy
└── processed.png

src/
├── real_world.jpg
├── real_world.npy
└── image.png

visualisations/
└── .gifs with varying scenarios


* settings.json
.gitignore
.gitattributes
README.md
```


- `simulate.py`: Main script for running the simulation.
- `settings.json`: Configuration file to customize simulation parameters.

## Configuration (settings.json)

Modify the `settings.json` file to control the simulation's behavior:

- **Grid Quality:**
  - `quality`: `"low"`, `"medium"`, or `"high"` for different grid resolutions. Corresponding to $100\times100$, $200\times200$ or $300\times300$ respectively.

- **Environmental Settings:**
  - `fire_probability`: Probability of fire igniting on each grid cell. If your simulation includes a plane, note that as your simulation quality increases this probability should decrease. Otherwise the plane will dart around with its head loose.
  - `has_rain`: `true`/`false` to include rainfall generated by Perlin noise.
  - `has_varying_densities`: `true`/`false` for 4 varied vegetation densities.
  - `burn_time` and `grow_time`: Arrays specifying burn and growth rates for 4 of the vegetation densities. If `has_varying_densities` is `false` then the simulation takes the first value from each list.
  - `wind_direction`: Options include `"N"`, `"S"`, `"E"`, `"W"`, `"NW"`, `"NE"`, `"SW"`, `"SE"`.
- **Image Settings:**
  - `use_personal_image`: `true`/`false` to load a custom grid from an image file.
  - `image_path`: Path to the custom image.
- **Fire Suppression:**
  - `has_plane`: `true`/`false` to enable or disable the firefighting plane.
  - `plane_speed`: The maximum speed of the plane in terms of cells per time step.
  - `plane_radius`: The effective radius of the plane for fire surpression.


## How to Run

### Personal Image
1. If you'd like to input your own area, head to this link and take a screenshot <a href="https://www.google.com/maps/d/viewer?mid=1OpMoz-v9iOYinQPbBzzx_lBT0QO8h-8&ll=-37.38159633507727%2C148.62546596105895&z=10" target="_blank">Forest Fires</a>.
2. Place your screenshot in the `src` directory and title it `image.png`
3. Follow the rest of the steps.

### Running the simulation
1. Make sure your configurations are as you'd like them in the `settings.json` file.
2. Go to `app/` and run `simulate.py`.
3. You should see two sets of plots confirming your simulation before it runs. One of the procesed image of your screenshot, the other of your new grid with rivers and a place with started fire.
4. Check for any inconsistencies that may cause errors in the simulation, for example if you've started a fire on top of a river.
5. View your simulation in `results/fire_sim.gif`.

HAPPY SIMULATING!