"""
Original source: https://github.com/neumicha/Blender2Helios, commit f109cdf043e602f7ea54c035162bbf7cb6b44131
(modified to be a standalone application, and adapted to my needs for exporting annotations)

NB: in the Blender file, all collections must be enabled and visible. Otherwise, bpy will raise:
    RuntimeError: Error: Object '...' can't be selected because it is not in View Layer 'RenderLayer'!
"""

import argparse
import math
import os
import bpy
from tqdm import tqdm

from shared_functions import abort_if_output_directory_not_empty, ensure_directory_exists


def convert_blender_file_to_helios_input(in_blender_file_path, output_dir_path):
    pref_sceneName = "blender2heliosScene"
    pref_alsoWriteSurveyFile = False
    pref_alwaysOverrideModels = True  # Overwrite output files when existing
    pref_useMaterials = False
    pref_useOwnMaterials = False
    pref_deleteCachedScene = True

    bpy.ops.wm.open_mainfile(filepath=in_blender_file_path)

    blender2heliosHelper = Blender2HeliosHelper(
        output_dir_path,
        pref_sceneName,
        pref_alsoWriteSurveyFile,
        pref_alwaysOverrideModels,
        pref_useMaterials,
        pref_useOwnMaterials,
        bpy.context.scene.cursor.location,
    )

    if pref_deleteCachedScene:
        blender2heliosHelper.deleteCachedScene()
    blender2heliosHelper.export2Helios()

    print("Done!")


class Blender2HeliosHelper:
    """Helper functions for Blender2Helios"""

    def __init__(self, output_dir_path, sceneName, alsoWriteSurveyFile, alwaysOverrideModels, useMaterials, useOwnMaterials, scannerLocation):
        self.heliosDir = output_dir_path
        self.sceneName = sceneName
        self.alsoWriteSurveyFile = alsoWriteSurveyFile
        self.alwaysOverrideModels = alwaysOverrideModels
        self.useMaterials = useMaterials
        self.useOwnMaterials = useOwnMaterials
        self.scannerLocation = scannerLocation

    def deleteCachedScene(self):
        sceneFile = os.path.join(self.heliosDir, "helios_input", "scenes", self.sceneName + ".scene")
        if os.path.exists(sceneFile):
            os.remove(sceneFile)

    def cutString(self, string, delim):
        if string.find(delim) == -1:
            return string
        else:
            return string[0 : string.find(delim)]

    def determine_semantic_class(self, object):
        semantic_class_candidates = []
        for face in object.data.polygons:
            # print("face", face.index, "material_index", face.material_index)
            slot = object.material_slots[face.material_index]
            if slot.material is not None:
                if slot.material.name not in semantic_class_candidates:
                    semantic_class_candidates.append(slot.material.name)
            else:
                # No material assigned to face
                semantic_class_candidates.append("unknown")

        if len(semantic_class_candidates) == 0:
            return "unknown"

        if len(semantic_class_candidates) == 1:
            return semantic_class_candidates[0]

        raise Exception(f"Object {object.name} has multiple semantic classes: {semantic_class_candidates}")
        # return ";".join(semantic_class_candidates)

    def buildSceneParts(self):
        out = ""
        part_id = 1
        for collection_num, collection in enumerate(bpy.data.collections):
            if collection.name != "Ignore":
                # TODO can we parallelize this and does bpy support this?
                for object in tqdm(collection.all_objects, desc=f'Exporting collection "{collection.name}" ({collection_num + 1}/{len(bpy.data.collections)})'):
                    # Do not export the curve object that is used for creating tunnel curvature. We don't have use for it.
                    if object.type == "CURVE":
                        continue

                    # print('-')
                    # print('Found object:', collection.name, '/', object.name)
                    objFileSizeExtension = self.dim2Text(self.dimScale2Original(object.dimensions, object.scale))
                    collectionDir = ensure_directory_exists(os.path.join(self.heliosDir, "helios_input", "sceneparts", collection.name))

                    objFile = collectionDir + "/" + object.name + "-" + objFileSizeExtension + ".obj"
                    scale = object.scale[0]
                    object.rotation_mode = "QUATERNION"  # Otherwise we only get zeros later
                    # export .obj file if needed
                    if not os.path.exists(objFile) or self.alwaysOverrideModels:
                        self.selectOneObject(object)
                        self.exportSelectedObject(objFile)
                        if self.useOwnMaterials:
                            self.prependMaterial2File(collection.name, objFile)

                    semantic_class = self.determine_semantic_class(object)

                    out += self.object2XML(
                        collection.name,
                        object.name + "-" + objFileSizeExtension + ".obj",
                        [0, 0, 0],  # already included in the exported .obj file, no need to let HELIOS++ rotate the part when loading
                        [0, 0, 0],  # already included in the exported .obj file, no need to let HELIOS++ translate the part when loading
                        scale,
                        part_id,
                        semantic_class,
                    )
                    part_id += 1
        return out

    def export2Helios(self):
        # Change to object mode
        bpy.ops.object.mode_set(mode="OBJECT", toggle=False)

        # Scene
        fScene = open(os.path.join(ensure_directory_exists(os.path.join(self.heliosDir, "helios_input", "scenes")), self.sceneName + ".xml"), "w+")
        fScene.write(self.xmlSceneHead())
        fScene.write(self.buildSceneParts())
        fScene.write(self.xmlSceneFoot())
        fScene.close()

        # Survey
        if self.alsoWriteSurveyFile:
            fSurvey = open(os.path.join(self.ensure_directory_exists(os.path.join(self.heliosDir, "helios_input", "surveys")), self.sceneName + ".xml"), "w+")
            fSurvey.write(self.xmlSurvey())
            fSurvey.close()

    # return the xml head of the scene

    def xmlSceneHead(self):
        return (
            """<?xml version="1.0" encoding="UTF-8"?>
    <document>
        <scene id=\""""
            + self.sceneName
            + """" name=\""""
            + self.sceneName
            + """">
    """
        )

    # returns xml code for survey (my be changed later for more scans etc.)

    def xmlSurvey(self):
        return (
            """<?xml version="1.0" encoding="UTF-8"?>
    <document>
        <!-- Default scanner settings: -->
        <scannerSettings id="profile1" active="true" pulseFreq_hz="100000" scanAngle_deg="50.0" scanFreq_hz="120" headRotatePerSec_deg="10.0" headRotateStart_deg="0.0" headRotateStop_deg="0.0" />
        <survey defaultScannerSettings="profile1" name=\""""
            + self.sceneName
            + """" scene=\""""
            + self.heliosDir
            + """/helios_input/scenes/"""
            + self.sceneName
            + """.xml#"""
            + self.sceneName
            + """" platform=\""""
            + self.heliosDir
            + """/helios_input/platforms.xml#tripod" scanner=\""""
            + self.heliosDir
            + """/helios_input/scanners_tls.xml#riegl_vz400">
            <leg>
                <platformSettings x=\""""
            + str(self.scannerLocation[0])
            + """" y=\""""
            + str(self.scannerLocation[1])
            + """" z=\""""
            + str(self.scannerLocation[2])
            + """" onGround="true" />
                <scannerSettings template="profile1" headRotateStart_deg="0" headRotateStop_deg="360" />
            </leg>
        </survey>
    </document>
    """
        )

    # Converts object to valid Helios XML part. Remember, that you have to do a PRY rotation in Helios (using RPY angles in degree)
    def object2XML(self, collection, objectFile, translation, rotation, scale, part_id: int, semantic_class: str):
        return (
            f"""        <part id=\"{part_id}\" semantic_class=\"{semantic_class}\">
                <filter type="objloader">
                    <param type="string" key="filepath" value=\""""
            + self.heliosDir
            + """/helios_input/sceneparts/"""
            + collection
            + "/"
            + objectFile
            + """" />
                    <param type="string" key="up" value="z" />
                </filter>
                <filter type="rotate">
                    <param type="rotation" key="rotation">
                        <rot axis="x" angle_deg=\""""
            + str(rotation[0])
            + """" />
                        <rot axis="y" angle_deg=\""""
            + str(rotation[1])
            + """"  />
                        <rot axis="z" angle_deg=\""""
            + str(rotation[2])
            + """"  />
                    </param>
                </filter>
                <filter type="translate">
                    <param type="vec3" key="offset" value=\""""
            + str(translation[0])
            + ";"
            + str(translation[1])
            + ";"
            + str(translation[2])
            + """" />
                </filter>
                <filter type="scale">
                    <param type="double" key="scale" value=\""""
            + str(scale)
            + """" />
                </filter>
            </part>
    """
        )

    # returns the xml footer of the scene
    def xmlSceneFoot(self):
        return """    </scene>
    </document>
    """

    # Brings first dimension to scale 1 and returns the 3 dimensions
    def dimScale2Original(self, dimensions, scale):
        return dimensions / scale[0]

    def dim2Text(self, dimensions):
        return str(int(dimensions[0] * 100)) + "-" + str(int(dimensions[1] * 100)) + "-" + str(int(dimensions[2] * 100))

    def exportSelectedObject(self, file):
        export_materials = self.useMaterials and not self.useOwnMaterials

        # Reference: https://docs.blender.org/api/current/bpy.ops.wm.html#bpy.ops.wm.obj_export
        bpy.ops.wm.obj_export(
            filepath=file,
            check_existing=False,
            export_selected_objects=True,
            export_normals=True,
            export_materials=export_materials,
            export_uv=False,
            apply_modifiers=True,
            forward_axis="Y",
            up_axis="Z",
            # filter_blender=False,
            # filter_backup=False,
            # filter_image=False,
            # filter_movie=False,
            # filter_python=False,
            # filter_font=False,
            # filter_sound=False,
            # filter_text=False,
            # filter_archive=False,
            # filter_btx=False,
            # filter_collada=False,
            # filter_alembic=False,
            # filter_usd=False,
            # filter_obj=False,
            # filter_volume=False,
            # filter_folder=True,
            # filter_blenlib=False,
            # filemode=8,
            # display_type='DEFAULT',
            # sort_method='DEFAULT',
            # export_animation=False,
            # start_frame=-2147483648,
            # end_frame=2147483647,
            # global_scale=1.0,
            # export_eval_mode='DAG_EVAL_VIEWPORT',
            # export_colors=False,
            # export_pbr_extensions=False,
            # path_mode='AUTO',
            # export_triangulated_mesh=False,
            # export_curves_as_nurbs=False,
            # export_object_groups=False,
            # export_material_groups=False,
            # export_vertex_groups=False,
            # export_smooth_groups=False,
            # smooth_group_bitflags=False,
            # filter_glob='*.obj;*.mtl'
        )

    def selectOneObject(self, object):
        bpy.ops.object.select_all(action="DESELECT")
        object.select_set(True)
        bpy.context.view_layer.objects.active = object

    # Quaternion (w,x,y,z) to Tiat Bryan (r,p,y); Output in degrees
    def quaternion2RPY(self, q):
        r = 180 / math.pi * math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (math.pow(q[1], 2) + math.pow(q[2], 2)))
        p = 180 / math.pi * math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
        y = 180 / math.pi * math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (math.pow(q[2], 2) + math.pow(q[3], 2)))
        return (r, p, y)

    def prependMaterial2File(self, materialName, fileName):
        # We read the existing text from file in READ mode
        src = open(fileName, "r")
        prepend = "mtllib ../materials.mtl\nusemtl " + materialName + "\n"  # Prepending string
        xml = src.readlines()
        # Here, we prepend the string we want to on first line
        xml.insert(0, prepend)
        src.close()
        # We again open the file in WRITE mode
        src = open(fileName, "w")
        src.writelines(xml)
        src.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Export Blender scenes to HELIOS++ input format.")
    parser.add_argument(
        "--scene",
        type=str,
        help="Run for a specific scene. If unset, run for all scenes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generated_tunnels_blender_path = os.path.join("helios_workspace", "generated_tunnels_blender")
    out_base_path = os.path.join("helios_workspace", "helios_surveys")

    blend_file_names = sorted([f for f in os.listdir(generated_tunnels_blender_path) if f.endswith(".blend")])

    if args.scene:
        wanted_scene_file_name = f"{args.scene}.blend"
        blend_file_names = [wanted_scene_file_name] if wanted_scene_file_name in blend_file_names else []

    if not blend_file_names:
        raise ValueError("No valid scenes found to process.")

    for blend_file_name in tqdm(blend_file_names):
        output_dir_path = os.path.join(out_base_path, blend_file_name.replace(".blend", ""))
        abort_if_output_directory_not_empty(output_dir_path)
        convert_blender_file_to_helios_input(os.path.join(generated_tunnels_blender_path, blend_file_name), output_dir_path)
