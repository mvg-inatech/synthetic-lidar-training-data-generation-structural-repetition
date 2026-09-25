import bpy
import sys
from tqdm import tqdm


def split_loose_parts():
    # Get all mesh objects in the scene
    objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    # Iterate over each object with a progress bar
    for obj in tqdm(objects, desc="Processing objects"):
        bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
        obj.select_set(True)  # Select the current object
        bpy.context.view_layer.objects.active = obj   # Set the active object

        # Enter Edit Mode
        bpy.ops.object.mode_set(mode='EDIT')

        # Select all mesh elements
        bpy.ops.mesh.select_all(action='SELECT')

        # Separate by loose parts
        bpy.ops.mesh.separate(type='LOOSE')

        # Return to Object Mode
        bpy.ops.object.mode_set(mode='OBJECT')

def main():
    input_file_path = "/data/helios/input.blend"
    output_file_path = "/data/helios/out.blend"

    # Open the input .blend file
    bpy.ops.wm.open_mainfile(filepath=input_file_path)

    # Perform the split operation
    split_loose_parts()

    # Save the processed .blend file
    bpy.ops.wm.save_as_mainfile(filepath=output_file_path)

if __name__ == "__main__":
    main()