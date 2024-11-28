# Forest Fires Industrial Mathematics

## Project Structure

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
├── * simulate.py
├── * image_processor.py
└── utils.py

results/
├── fire_sim.gif
├── processed.npy
└── processed.png

src/
└── image.png

.gitignore
.gitattributes
README.md
```

## Features
### Input Your Own Area!
Go to <a href="https://www.google.com/maps/d/viewer?mid=1OpMoz-v9iOYinQPbBzzx_lBT0QO8h-8&ll=-37.38159633507727%2C148.62546596105895&z=10" target="_blank">Forest Fires</a> and take a screenshot of an area of interest. Then follow the suggested workflow and simulate a fire wherever you'd like!

### Planes ✈️ !
In `simulate.py` you can initiate a plane at any starting block you'd like and watch it try and stop the fire.

Disclaimer: our clustering algorithm is a very simple DFS algorithm that does not prioritise things like distance to fire and potential future damage. These are some future extensions we could take a look at.


## Suggested Workflow

1. Go to <a href="https://www.google.com/maps/d/viewer?mid=1OpMoz-v9iOYinQPbBzzx_lBT0QO8h-8&ll=-37.38159633507727%2C148.62546596105895&z=10" target="_blank">Forest Fires</a> and take a screenshot of an area.

2. Label your screenshot 'image.png' and input into `src` folder.

3. In the `app` folder run `image_processor.py`. You should see a plot of your processed image. Additionally, you can check your processed image `processed.png` in the `src` folder along with a `processed.npy` file that will be used to run the simulation.

4. Finally, to run the simulation go into the `app` folder and run `simulat.py`

note: within `simulate.py` there are more things you can tweak, which we will explore below.

## Plane Algorithm

## Fire Algorithm