# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import mujoco

from pick_and_place import build_scene, export_scene


def test_scene_contains_robot_floor_light_and_cube():
    model = build_scene().compile()

    assert model.body("base").id >= 0
    assert model.body("pick_cube").id >= 0
    assert model.nbody == build_scene(wrist_camera=False).compile().nbody + 2
    floor = model.geom("floor").id
    assert model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE
    assert tuple(model.geom_size[floor, :2]) == (0.0, 0.0)
    assert model.mat(model.geom_matid[floor]).name == "groundplane"
    assert model.texture("groundplane").id >= 0
    assert model.light("scene_light").id >= 0
    assert model.nlight == 1
    assert model.body_jntnum[model.body("pick_cube").id] == 0
    assert tuple(model.geom_size[model.geom("pick_cube").id]) == (0.015, 0.015, 0.015)


def test_export_scene_writes_compilable_xml(tmp_path):
    output = export_scene(tmp_path / "scene.xml")

    assert output == tmp_path / "scene.xml"
    assert output.exists()
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.body("pick_cube").id >= 0
