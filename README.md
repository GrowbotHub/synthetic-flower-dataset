# Synthetic Dataset Creation

Create synthetic dataset of 3D meshes loaded into Pyrender using Trimesh.

## Getting Started

### Prerequisites

The librairies needed can be installed by creating a new environment with:
```
conda create --name synthetic_ds opencv numpy -y
```
Then 
```
conda activate synthetic_ds
```
And finally
```
pip install pyrender
```

If the following error appears when running the code
```
AttributeError: module 'pyglet.gl' has no attribute 'xlib'
```
One needs to modify line 52 in `pyglet_platform.py`, change 

```
   def make_uncurrent(self):
       import pyglet
       pyglet.gl.xlib.glx.glXMakeContextCurrent(self._window.context.x_display, 0, 0, None)
```
To 
```
    def make_uncurrent(self):
        try:
            import pyglet.gl.xlib
            pyglet.gl.xlib.glx.glXMakeContextCurrent(self._window.context.x_display, 0, 0, None)
        except:
            pass
```
Additionally, if the following error appears
```
ImportError: ('Unable to load OpenGL library', 'dlopen(OpenGL, 10): image not found', 'OpenGL', None)
```
see [here](https://github.com/PixarAnimationStudios/USD/issues/1372#issuecomment-716925973) for a solution.

## Running the code

The main file is `setup.py`. It will load a synthetic mesh into Pyrender with the use of Trimesh. Configurations such as pose of the camera and lighting conditions can be specified before the flower mesh is presented in a Pyrender scene. Parameters related to generating images can also be specified such as the number of different camera poses to use. The user will then be asked whether to generate the a batch of images or not. If yes, the images and an associated CSV file are created. If no, the program is terminated.

The outputs get saved to the `outputs` subdirectory and all the synthetic models are in the `models` subdirectory.

## Authors

* **Gil Tinde** 


## Acknowledgments

* [Pyrender](https://pyrender.readthedocs.io/en/latest/)
* [Trimesh](https://trimsh.org/)
