from typing import Tuple
import bpy
import random
from math import radians
from mathutils import Vector, Euler, Quaternion  # type: ignore
import argparse


def determine_segment_length_and_extent(collection):
    """Determine the segment length and Y extent of the input tunnel slice."""
    min_y = float("inf")
    max_y = float("-inf")
    for obj in collection.objects:
        obj_matrix = obj.matrix_world
        for corner in obj.bound_box:
            world_corner = obj_matrix @ Vector(corner)
            min_y = min(min_y, world_corner.y)
            max_y = max(max_y, world_corner.y)
    segment_length = max_y - min_y
    return segment_length, min_y, max_y


def object_spans_entire_slice(obj, slice_min_y, slice_max_y, tolerance_meters=0.1):
    """Determine if the object spans the entire Y extent of the tunnel slice."""
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    coords = [v.y for v in bbox]
    min_coord = min(coords)
    max_coord = max(coords)
    if (abs(min_coord - slice_min_y) < tolerance_meters) and (abs(max_coord - slice_max_y) < tolerance_meters):
        return True
    else:
        return False


def apply_transforms(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def process_continuous_object(obj, path_obj, generated_collection, segment_length, slice_min_y, slice_max_y):
    """Process and replicate an object along the path with correct spacing."""
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    new_obj.animation_data_clear()
    generated_collection.objects.link(new_obj)  # add object to output collection

    # Required to make curve modifier work correctly (otherwise it will not follow the path correctly and be located somewhere else)
    apply_transforms(new_obj)  # Apply location transform (there should not be any rotation transform yet)
    new_obj.rotation_euler = Euler((0, radians(90), 0), "XYZ")
    apply_transforms(new_obj)  # Apply rotation transform (there is no location transform anymore)

    # Add Subdivision Surface modifier to smooth the wall curvature
    subsurf_modifier = new_obj.modifiers.new("Subsurf", "SUBSURF")
    subsurf_modifier.levels = 4
    subsurf_modifier.render_levels = 4
    subsurf_modifier.subdivision_type = "SIMPLE"
    subsurf_modifier.quality = 1  # Speed-up calculation

    # Optionally, add edge creases to preserve sharp edges
    bpy.context.view_layer.objects.active = new_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.transform.edge_crease(value=1.0)
    bpy.ops.object.mode_set(mode="OBJECT")

    array_modifier = new_obj.modifiers.new("TunnelArray", "ARRAY")
    array_modifier.fit_type = "FIT_CURVE"
    array_modifier.curve = path_obj
    array_modifier.use_merge_vertices = True
    array_modifier.use_merge_vertices_cap = True

    # Determine the offset for the array modifier
    obj_length = obj.dimensions.y
    if obj_length == 0:
        obj_length = 0.001  # Avoid division by zero
    if object_spans_entire_slice(obj, slice_min_y, slice_max_y):
        # Spans entire Y extent, normal offset
        array_modifier.relative_offset_displace = (0, 1, 0)
    else:
        # Does not span entire Y extent, adjust offset
        offset = segment_length / obj_length
        array_modifier.relative_offset_displace = (0, offset, 0)

    curve_modifier = new_obj.modifiers.new("TunnelCurve", "CURVE")
    curve_modifier.object = path_obj
    curve_modifier.deform_axis = "POS_Y"


def process_discrete_object(obj, path_obj, generated_collection, segment_length, num_segments):
    """Process and replicate an object for each tunnel segment."""
    obj_length = obj.dimensions.y
    if obj_length == 0:
        obj_length = 0.001  # Avoid division by zero

    for i in range(num_segments + 1):
        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        new_obj.animation_data_clear()

        # Calculate the position for each object
        offset_position = i * segment_length

        generated_collection.objects.link(new_obj)  # add object to output collection

        # Required to make curve modifier work correctly (otherwise it will not follow the path correctly and be located somewhere else)
        apply_transforms(new_obj)
        new_obj.rotation_euler = Euler((0, radians(90), 0), "XYZ")

        # Set the new location
        new_obj.location = Vector((0, offset_position, 0))

        # Add curve modifier to follow the path
        curve_modifier = new_obj.modifiers.new("TunnelCurve", "CURVE")
        curve_modifier.object = path_obj
        curve_modifier.deform_axis = "POS_Y"


def repeat_input_slice_objects_along_curve(
    tunnel_slice_collection,
    path_obj,
    generated_collection,
    segment_length,
    slice_min_y,
    slice_max_y,
    num_segments,
):
    """Process all objects in the tunnel slice collection."""
    for obj in tunnel_slice_collection.objects:
        if object_spans_entire_slice(obj, slice_min_y, slice_max_y):
            process_continuous_object(
                obj,
                path_obj,
                generated_collection,
                segment_length,
                slice_min_y,
                slice_max_y,
            )
        else:
            process_discrete_object(obj, path_obj, generated_collection, segment_length, num_segments)


def remove_collection(collection):
    """Remove a collection from the scene."""
    bpy.data.collections.remove(collection)


def optimize_file_size():
    """Optimize file size by removing unused data blocks."""
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def save_tunnel(save_path):
    """Save the generated tunnel to disk."""
    bpy.ops.wm.save_as_mainfile(filepath=save_path)


def load_clean_workspace():
    """Loads a clean workspace."""
    bpy.ops.wm.read_homefile(use_empty=True)


def create_generated_tunnel_collection():
    """Create a new collection for the generated tunnel."""
    generated_collection = bpy.data.collections.new("GeneratedTunnel")
    bpy.context.scene.collection.children.link(generated_collection)
    return generated_collection


def remove_default_collection():
    """Remove the default collection 'Collection'."""
    collection_to_remove = bpy.data.collections.get("Collection")
    if collection_to_remove:
        bpy.context.scene.collection.children.unlink(collection_to_remove)
        bpy.data.collections.remove(collection_to_remove)


def load_tunnel_slice(tunnel_slice_path, tunnel_slice_collection_name):
    """Load the tunnel slice Blender file and return the collection."""
    bpy.ops.wm.append(
        filename=tunnel_slice_collection_name,
        directory=tunnel_slice_path + "/Collection/",
    )
    return bpy.data.collections[tunnel_slice_collection_name]


def create_straight_tunnel_path_curve(num_segments, segment_length):
    """Create a straight curve for the tunnel path."""
    curve_data = bpy.data.curves.new("TunnelPath", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 12
    path_obj = bpy.data.objects.new("TunnelPath", curve_data)
    spline = curve_data.splines.new("POLY")
    spline.points.add(num_segments)
    for i in range(num_segments + 1):  # +1 because we need one more point than segments
        position = Vector((0, i * segment_length, 0))
        spline.points[i].co = (position.x, position.y, position.z, 1)
    return path_obj


def add_curvature_to_tunnel_path_curve(
    path_obj,
    curve_probability,
    elevation_change_probability,
    curve_angle_range,
    elevation_angle_range,
):
    """Modify the path curve to add curves and elevation changes."""
    curve_data = path_obj.data
    spline = curve_data.splines[0]

    # Convert the spline to a Bezier spline for smoother control
    poly_points = spline.points
    positions = [Vector((point.co.x, point.co.y, point.co.z)) for point in poly_points]
    curve_data.splines.remove(spline)
    bezier_spline = curve_data.splines.new(type="BEZIER")
    bezier_spline.bezier_points.add(len(positions) - 1)

    for i, point in enumerate(positions):
        bp = bezier_spline.bezier_points[i]
        bp.co = point
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"

    # Apply random curves and elevation changes
    total_rotation = Quaternion()
    for i in range(1, len(bezier_spline.bezier_points)):
        is_curve = random.random() < curve_probability
        is_elevation_change = random.random() < elevation_change_probability
        curve_angle = radians(random.uniform(*curve_angle_range)) if is_curve else 0
        elevation_angle = radians(random.uniform(*elevation_angle_range)) if is_elevation_change else 0

        # Create delta rotations as quaternions
        delta_rot_curve = Quaternion((0, 0, 1), curve_angle)  # Rotation around Z-axis
        delta_rot_elevation = Quaternion((1, 0, 0), elevation_angle)  # Rotation around X-axis

        # Combine rotations
        delta_rotation = delta_rot_curve @ delta_rot_elevation
        total_rotation = total_rotation @ delta_rotation

        direction = total_rotation @ Vector((0, (positions[i] - positions[i - 1]).length, 0))

        previous_point = bezier_spline.bezier_points[i - 1].co
        current_point = previous_point + direction
        bezier_spline.bezier_points[i].co = current_point

    # Update handles
    for bp in bezier_spline.bezier_points:
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"

    spline_points = []
    for bp in bezier_spline.bezier_points:
        spline_points.append((bp.co.x, bp.co.y, bp.co.z))
    return spline_points


def apply_all_modifiers(generated_collection):
    """Apply modifiers for all objects in the generated collection."""
    for obj in generated_collection.objects:
        bpy.context.view_layer.objects.active = obj
        for modifier in obj.modifiers:
            bpy.ops.object.modifier_apply(modifier=modifier.name)


def export_tunnel_metadata(metadata, output_path):
    """Export metadata about the generated tunnel."""
    metadata_path = bpy.path.abspath(output_path)
    with open(metadata_path, "w") as f:
        import json

        json.dump(metadata, f, indent=4)


def generate_long_tunnel(
    input_tunnel_slice_path: str,
    output_long_tunnel_save_path: str,
    input_tunnel_slice_collection_name: str,
    rng_seed: int = 42,
    num_segments: int = 50,
    curve_probability: float = 0.4,
    elevation_change_probability: float = 0.2,
    curve_angle_range: Tuple[float, float] = (-30, 30),
    elevation_angle_range: Tuple[float, float] = (-10, 10),
    apply_modifiers: bool = False,
):
    """Main function to generate a synthetic long tunnel."""
    # Initialize random number generator
    random.seed(rng_seed)

    load_clean_workspace()
    input_tunnel_slice_collection = load_tunnel_slice(input_tunnel_slice_path, input_tunnel_slice_collection_name)
    segment_length, slice_min_y, slice_max_y = determine_segment_length_and_extent(input_tunnel_slice_collection)
    generated_tunnel_collection = create_generated_tunnel_collection()
    remove_default_collection()

    path_obj = create_straight_tunnel_path_curve(num_segments, segment_length)
    generated_tunnel_collection.objects.link(path_obj)
    curve_points = add_curvature_to_tunnel_path_curve(
        path_obj,
        curve_probability,
        elevation_change_probability,
        curve_angle_range,
        elevation_angle_range,
    )

    export_tunnel_metadata(
        {
            "seed": rng_seed,
            "slice_type": input_tunnel_slice_collection_name,
            "segment_length": segment_length,
            "slice_min_y": slice_min_y,
            "slice_max_y": slice_max_y,
            "num_segments": num_segments,
            # The curve path starts at the end of the first segment and ends at the end of the last segment. The path will be used as the trajectory for the LiDAR simulator. Leave one segment before and after the trajectory to give some surrounding context info.
            "curve_points": curve_points[:-1],
        },
        output_long_tunnel_save_path.replace(".blend", "_metadata.json"),
    )

    repeat_input_slice_objects_along_curve(
        input_tunnel_slice_collection,
        path_obj,
        generated_tunnel_collection,
        segment_length,
        slice_min_y,
        slice_max_y,
        num_segments,
    )

    if apply_modifiers:
        # Apply modifiers after the path has been modified. Increases file size and saves computation time later.
        apply_all_modifiers(generated_tunnel_collection)

    remove_collection(input_tunnel_slice_collection)
    bpy.ops.object.select_all(action="DESELECT")  # Deselect all objects

    optimize_file_size()
    save_tunnel(output_long_tunnel_save_path)
    print("Tunnel generation complete. File saved to:", output_long_tunnel_save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a long tunnel from a tunnel slice.")
    parser.add_argument(
        "--input_tunnel_slice_path", type=str, default="helios_workspace/10 m slices (step 3 - split instances).blend", help="Path to the tunnel slice file"
    )
    parser.add_argument(
        "--input_tunnel_slice_collection_name", type=str, default="tunnel slice - type 6 - emergency exit", help="Name of the tunnel slice collection"
    )
    parser.add_argument(
        "--output_long_tunnel_save_path", type=str, default="helios_workspace/generated_tunnel.blend", help="Path to save the generated long tunnel"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed that determines the tunnel appearance")
    parser.add_argument("--num_segments", type=int, default=5, help="Number of segments in the long tunnel")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_long_tunnel(
        input_tunnel_slice_path=args.input_tunnel_slice_path,
        input_tunnel_slice_collection_name=args.input_tunnel_slice_collection_name,
        output_long_tunnel_save_path=args.output_long_tunnel_save_path,
        rng_seed=args.seed,
        num_segments=args.num_segments,
    )
