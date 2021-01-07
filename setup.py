"""
Setup pyrender scene for generating dataset.
"""
import numpy as np
import trimesh
from pyrender import PerspectiveCamera,\
                     PointLight,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags
from PIL import Image
from cameraCharacteristics import CameraPose, CameraDistance
import random
from transformations import *
from generate_dataset import generate
from flower_utils import get_flower_pose,\
                         get_flower_mesh_indices

# A few colors
yellowRGB = [236, 228, 53]
orchidRGB = [153,50,204]
lettuceGreenRGB = [2, 173, 7]
redRGB = [255, 0, 0]

# Types of flowers that either don't have a center or stem sub-mesh
no_center_flowers = ['celery', 'cantaloupe', 'cucumber']
no_stem_flowers = ['chili_pepper', 'watermelon', 'cantaloupe', 
                   'cucumber', 'cucurbita_maxima', 'zucchini', 
                   'sunflower', 'bean', 'peas', 'cherry_tomato',
                   'eggplant', 'tea', 'strawberry', 'cotton',
                   'tobacco', 'crocus', 'dahlia', 'gardenia',
                   'spider_gerbera', 'hibiscus', 'lotus', 'orchid',
                   'peony', 'passionflower', 'african_violet',
                   'bat_flower', 'marigold']

#==============================================================================
# Mesh creation
#==============================================================================

file_to_type = {
'FL22_1.obj': 'poppy_anemone', 'FL22_2.obj': 'poppy_anemone', 'FL22_3.obj': 'poppy_anemone', 
'FL22_4.obj': 'poppy_anemone', 'FL22_5.obj': 'poppy_anemone',
'FL32_1.obj': 'gerbera', 'FL32_2.obj': 'gerbera', 'FL32_3.obj': 'gerbera',
'VG05_3.obj': 'capitata', 'VG05_4.obj': 'capitata' , 'VG05_7.obj': 'capitata',
'VG01_7.obj': 'onion', 'VG01_8.obj': 'onion', 'VG01_9.obj': 'onion', 
'VG02_8.obj': 'celery', 
'VG04_2.obj': 'italica', 
'VG06_4.obj': 'cauliflower', 
'VG07_9.obj': 'chili_pepper',
'VG08_2.obj': 'watermelon',
'VG09_6.obj': 'cantaloupe', 'VG09_9.obj': 'cantaloupe',
'VG10_3.obj': 'cucumber',
'VG11_3.obj': 'cucurbita_maxima',
'VG12_6.obj': 'zucchini', 'VG12_7.obj': 'zucchini',
'VG14_4.obj': 'sunflower',
'VG16_6.obj': 'bean',
'VG17_6.obj': 'peas',
'VG18_1.obj': 'cherry_tomato',
'VG19_3.obj': 'eggplant',
'AG02_9.obj': 'tea',
'AG06_5.obj': 'strawberry', 'AG06_9.obj': 'strawberry',
'AG08_5.obj': 'cotton', 'AG08_6.obj': 'cotton', 
'AG13_7.obj': 'tobacco', 'AG13_7.glb': 'tobacco',
'FL43_1.glb': 'crocus', 'FL43_2.glb': 'crocus', 'FL43_6.glb': 'crocus', 'FL43_7.glb': 'crocus', 
'FL44_1.glb': 'dahlia', 'FL44_9.glb': 'dahlia',
'FL45_2.glb': 'gardenia',
'FL46_1.glb': 'spider_gerbera', 'FL46_4.glb': 'spider_gerbera', 'FL46_6.glb': 'spider_gerbera', 
'FL46_7.glb': 'spider_gerbera', 'FL46_9.glb': 'spider_gerbera',
'FL47_1.glb': 'hibiscus', 'FL47_4.glb': 'hibiscus', 'FL47_7.glb': 'hibiscus',
'FL49_2.glb': 'lotus', 'FL49_4.glb': 'lotus', 'FL49_7.glb': 'lotus',
'FL50_1.glb': 'orchid', 'FL50_4.glb': 'orchid', 'FL50_9.glb': 'orchid',
'FL51_3.glb': 'peony', 'FL51_8.glb': 'peony', 
'FL52_1.glb': 'passionflower', 'FL52_5.glb': 'passionflower', 'FL52_6.glb': 'passionflower', 
'FL54_1.glb': 'african_violet', 'FL54_4.glb': 'african_violet', 'FL54_5.glb': 'african_violet', 
'FL54_7.glb': 'african_violet',
'FL56_1.glb': 'bat_flower',
'FL57_2.glb': 'marigold', 'FL57_5.glb': 'marigold', 'FL57_6.glb': 'marigold' 
} 

# User specifies filename
file_name = 'FL57_6.glb'

# Specify if decor shiuld be added
add_decor = True

# Get the flower_type based on the filename
flower_type = file_to_type[file_name]

# Determine path name
path_name = './models/flower_models/obj/'+flower_type+'/'

# Load the obj file
flower_trimesh = trimesh.load(path_name+file_name)

# All meshes used default to Scenes. We use dump() to 
# retrieve a list of individual sub-meshes (Trimesh)
if (type(flower_trimesh) == trimesh.scene.scene.Scene):
    trimesh_list = flower_trimesh.dump()
else:
    trimesh_list = [flower_trimesh]

# Convert from Trimesh to Mesh
mesh_list = []
for tri_mesh in trimesh_list:
    mesh_list.append(Mesh.from_trimesh(tri_mesh))

# Get indices of flower_mesh and center_mesh in the mesh_list
index_1, index_2, index_3 = get_flower_mesh_indices(file_name)

# Different options for the base/ceiling
base_types = ['soil','plain','ceiling','hydroponic']
base_type = base_types[0]
ceiling_type = base_types[2]
hydroponic_type = base_types[3]

# Base mesh
base_trimesh = trimesh.load('./models/scene/'+base_type+'_base.obj')
base_mesh = Mesh.from_trimesh(base_trimesh)

# Alternative base mesh
hydroponic_base_trimesh = trimesh.load('./models/scene/'+hydroponic_type+'_base.obj')
hydroponic_base_mesh = Mesh.from_trimesh(hydroponic_base_trimesh)

# Inf base mesh (big square that sits underneath the scene)
inf_base_trimesh = trimesh.load('./models/scene/'+base_type+'_base.obj')
inf_base_mesh = Mesh.from_trimesh(base_trimesh)

# Ceiling mesh 
ceiling_trimesh = trimesh.load('./models/scene/'+ceiling_type+'_base.obj')
ceiling_mesh = Mesh.from_trimesh(ceiling_trimesh)

# Different options for the walls
wall_types = ['hydroponic_farm', 'other']
wall_type = wall_types[0]

# Side "walls" mesh
wall_trimesh = trimesh.load('./models/scene/'+wall_type+'_wall.obj')
wall_mesh = Mesh.from_trimesh(wall_trimesh)

# Decor mesh is simply the leafy part from the cauliflower mesh 
# which can be added if we want more leaves in the background
decor_file_name = 'VG06_4.obj'
decor_type = file_to_type[decor_file_name]
decor_path_name = './models/flower_models/obj/'+decor_type+'/'
decor_trimesh = trimesh.load(decor_path_name+decor_file_name)
decor_trimesh_list = decor_trimesh.dump()
decor_mesh = Mesh.from_trimesh(decor_trimesh_list[0])
decor_pose = get_flower_pose(decor_type, decor_file_name)

#==============================================================================
# Light creation
#==============================================================================

# Lights for the walls
side_intensity = 2.0
spot_l_sides = PointLight(color=np.ones(3), intensity=side_intensity)

# Light for the flower pointing down at origin
point_l = PointLight(color=np.ones(3), intensity=0.4)

# Light for the flower
min_intensity = 0.05
max_intensity = 0.2
number_of_intensities = 5
varying_light_intensities = np.linspace(min_intensity,max_intensity,number_of_intensities).tolist()
point_l_cam = PointLight(color=np.ones(3), intensity=max_intensity)

#==============================================================================
# Pose creation
#==============================================================================

# Bases and ceiling pose
base_pose = spatial_transform_Matrix(scale = 1/2, t_z=0)
hydroponic_base_pose = spatial_transform_Matrix(scale = 1/4, t_z=0.001)
inf_base_pose = spatial_transform_Matrix(scale = 4, t_z=-1)
ceiling_pose = spatial_transform_Matrix(scale = 1, t_z=1, roll=np.pi)

# Side wall poses
side_closeness = 1
side_height = 0
side_E_pose = spatial_transform_Matrix(roll=np.pi/2, yaw=np.pi/2, t_x=-1/side_closeness, t_z=side_height)
side_N_pose = spatial_transform_Matrix(roll=np.pi/2, t_y=1/side_closeness, t_z=side_height)
side_W_pose = spatial_transform_Matrix(roll=np.pi/2, yaw=-np.pi/2, t_x=1/side_closeness, t_z=side_height)
side_S_pose = spatial_transform_Matrix(roll=np.pi/2, yaw=np.pi, t_y=-1/side_closeness, t_z=side_height)
wall_poses = [side_E_pose, side_N_pose, side_W_pose, side_S_pose]

# Poses for lights for the walls
point_l_E = spatial_transform_Matrix(roll=np.pi/12, yaw=np.pi/2, t_x=-1/((side_closeness)+0.5),t_z=side_height+1/8)
point_l_N = spatial_transform_Matrix(roll=np.pi/12, t_y=1/((side_closeness)+0.5),t_z=side_height+1/8)
point_l_W = spatial_transform_Matrix(roll=np.pi/12, yaw=-np.pi/2, t_x=1/((side_closeness)+0.5),t_z=side_height+1/8)
point_l_S = spatial_transform_Matrix(roll=np.pi/12, yaw=np.pi, t_y=-1/((side_closeness)+0.5),t_z=side_height+1/8)
wall_light_poses = [point_l_E, point_l_N, point_l_W, point_l_S]

# Pose for light on flower
light_pose, _ = lookAt(view=CameraPose.TOP, distance=CameraDistance.MEDIUM)

# Flower pose
flower_pose = get_flower_pose(flower_type, file_name)

#==============================================================================
# Scene creation
#==============================================================================

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

#==============================================================================
# Adding objects to the scene
#==============================================================================

# Flower
flower_mesh = mesh_list.pop(index_1)    
flower_node = scene.add(flower_mesh, pose=flower_pose, name='flower')

# Center (aka stigma)
if flower_type in no_center_flowers:
    center_mesh = flower_mesh
else:
    center_mesh =  mesh_list.pop(index_2)
    center_node = scene.add(center_mesh, pose=flower_pose, name='center')

# Stem (often not used)
stem_mesh = mesh_list.pop(index_3)
stem_node = scene.add(stem_mesh, pose=flower_pose, name='stem')

# The other sub-meshes of the flowers
for i in range(len(mesh_list)):
    scene.add(mesh_list[i], pose=flower_pose, name='flower'+str(i))

# Base square, inf square and walls
base_node = scene.add(base_mesh, pose=base_pose)
hydroponic_base_node = scene.add(hydroponic_base_mesh, pose=hydroponic_base_pose, name='hydroponic')
inf_base_node = scene.add(inf_base_mesh, pose=inf_base_pose)
ceiling_node = scene.add(ceiling_mesh, pose=ceiling_pose)
for pose in wall_poses:
    scene.add(wall_mesh, pose=pose, name='wall')

# Decor (some simple green leaves on the z = 0 plane)
if add_decor:
    decor_node = scene.add(decor_mesh, pose=decor_pose, name='decor')

#==============================================================================
# Adding lights
#==============================================================================

# Add light for flower
point_l_node = scene.add(point_l_cam, pose=light_pose, name='varying_light')

# Add lights for walls
for wall_light_pose in wall_light_poses:
    scene.add(spot_l_sides, pose=wall_light_pose, name='wall_light')

#==============================================================================
# Camera creation
#==============================================================================

# Camera properties
yfov = (np.pi / 3.0)
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600
aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT

# Camera
camera = PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio, znear=0.01)

# Specify parameters of camera of scene that will be previewed
alpha, beta, distance = 0, np.pi/3, 0.06

# Compute centroid of flower (3D mesh coordinates)
centroid = center_mesh.centroid

# Compute the edge of the flower (3D mesh coordinates)
flower_edge = random.choice(flower_mesh.bounds)

# Compute at the average of centroid and edge (3D mesh coordinates)
centroid_edge = [0] * 3
flower_edge_away = [0] * 3
for i in range(3):
    centroid_edge[i] = (centroid[i] + flower_edge[i])/2.0
    flower_edge_away[i] = (flower_edge[i] - centroid[i]/6.0)

# Choose with of the four points to point at
look_at = flower_edge_away

# Compute 3D coordinates of chosen point (3D world coordinates)
look_at_formatted = np.hstack((look_at, np.array([1]))).T
look_at_3D = ((flower_pose @  look_at_formatted).T)
at_x, at_y, at_z = look_at_3D[0], look_at_3D[1], look_at_3D[2]

# Or point at the origin
# at_x, at_y, at_z = 0,0,0

# Create camera pose and add it to the scene
new_campose, _ = lookAt(distance=distance, alpha=alpha, beta=beta, at_x=at_x, at_y=at_y, at_z=at_z)
cam_node = scene.add(camera, pose=new_campose)

# A light pointing from the camera's perspective could also be added 
# point_l_node = scene.add(point_l_cam, pose=new_campose)

#==============================================================================
# Use viewer to display scene
#==============================================================================

# View the scene
view_render_flags = {'cull_faces': False, 'vertex_normals': False}
v = Viewer(scene, render_flags=view_render_flags, viewport_size=(IMAGE_WIDTH, IMAGE_HEIGHT))

#==============================================================================
# Prepare for dataset generation
#==============================================================================

flags = RenderFlags.SKIP_CULL_FACES
r = OffscreenRenderer(viewport_width=IMAGE_WIDTH, viewport_height=IMAGE_HEIGHT)
# color, _ = r.render(scene, flags=flags)

# Generate dataset ?
choice = input("Generate dataset? [y/n]: ")
accepted_inputs = ['y','n']
while (not (choice in accepted_inputs)):
    choice = input("Generate dataset? [y/n]: ")

iterations = 5
mode = CameraPose.NONE
csvfile = './outputs/flowers_dataset.csv'
csvMode = 'w' # 'a' => append / 'w' => write (overwrites existing csv file)
meshes = [(flower_mesh, 'flower'), (stem_mesh, 'stem'), (center_mesh, 'center')]
skip_default_view = False
salt = '001'
offset = 0
skip_sub_mesh = {name: False for (mesh, name) in meshes}

if flower_type in no_stem_flowers:
    skip_sub_mesh['stem'] = True

if choice == 'y':

    # Remove camera that was used as preview from the scene
    scene.remove_node(cam_node)

    generate(
        scene=scene, renderer=r, flags=flags, alpha=alpha, beta=beta, camera=camera, 
        iterations=iterations, flower_pose=flower_pose, meshes=meshes, 
        light_intensitites = varying_light_intensities,
        skip_sub_mesh= skip_sub_mesh, at_x=at_x, at_y=at_y, at_z=at_z,
        skip_default_view=skip_default_view, offset=offset, 
        distance=distance, mode=mode, csvfile=csvfile, csvMode=csvMode,
        flower_name=file_name, salt=salt
        )

print("Finished")