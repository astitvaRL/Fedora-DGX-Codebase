

To get running, first start pose server:

````bash
conda activate animated_drawingsRL
cd /Users/hjessmith/Projects/AnimatedDrawingsRL/bvh_frame_streamer
python server.py
````

Then run AD:
````bash
conda activate animated_drawingsRL
cd /Users/hjessmith/Projects/AnimatedDrawingsRL/animated_drawings
python render.py /Users/hjessmith/Projects/AnimatedDrawingsRL/bvh_frame_streamer/development_config.yaml
````