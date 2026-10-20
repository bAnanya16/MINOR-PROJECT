"""Builds low-poly, realistic glTF/GLB models for the 3D warehouse viewer:
- shelf.glb: 1m x 1.8m x 1m industrial warehouse racking
- robot.glb: ~0.75m low warehouse AMR drive unit with a designated 'LED' material
- barrier.glb: Safety road barrier / construction cone with warning stripes
"""

import os
import trimesh
import numpy as np


def create_shelf_glb(out_path: str):
    scene = trimesh.Scene()
    
    # Industrial rack uprights (4 corner posts: 0.05m x 1.8m x 0.05m)
    # Total footprint: 1.0m width (x), 1.0m depth (z), 1.8m height (y)
    blue_color = [30, 80, 180, 255]
    orange_color = [230, 95, 20, 255]
    gray_color = [100, 105, 115, 255]
    
    w, h, d = 0.96, 1.8, 0.96
    post_th = 0.04
    
    # 4 corner upright posts
    for x_sign in [-1, 1]:
        for z_sign in [-1, 1]:
            post = trimesh.creation.box(extents=(post_th, h, post_th))
            px = x_sign * (w / 2 - post_th / 2)
            pz = z_sign * (d / 2 - post_th / 2)
            py = h / 2
            post.apply_translation([px, py, pz])
            post.visual.face_colors = blue_color
            scene.add_geometry(post, node_name=f"post_{x_sign}_{z_sign}")
            
    # 3 horizontal shelving beams / levels (at y=0.1, y=0.9, y=1.75)
    for y_lvl in [0.12, 0.92, 1.74]:
        # Cross beams along X
        for z_sign in [-1, 1]:
            beam = trimesh.creation.box(extents=(w, 0.06, 0.03))
            beam.apply_translation([0, y_lvl, z_sign * (d / 2 - 0.015)])
            beam.visual.face_colors = orange_color
            scene.add_geometry(beam, node_name=f"beam_x_{y_lvl}_{z_sign}")
        # Cross beams along Z
        for x_sign in [-1, 1]:
            beam_z = trimesh.creation.box(extents=(0.03, 0.06, d - 0.06))
            beam_z.apply_translation([x_sign * (w / 2 - 0.015), y_lvl, 0])
            beam_z.visual.face_colors = orange_color
            scene.add_geometry(beam_z, node_name=f"beam_z_{y_lvl}_{x_sign}")
            
        # Shelf decking panel
        deck = trimesh.creation.box(extents=(w - 0.04, 0.02, d - 0.04))
        deck.apply_translation([0, y_lvl + 0.03, 0])
        deck.visual.face_colors = gray_color
        scene.add_geometry(deck, node_name=f"deck_{y_lvl}")

    # Add sample storage boxes on the shelves
    box_color1 = [180, 140, 95, 255]   # Cardboard
    box_color2 = [160, 120, 80, 255]
    
    # 2 boxes on lower shelf
    b1 = trimesh.creation.box(extents=(0.35, 0.35, 0.35))
    b1.apply_translation([-0.2, 0.15 + 0.175 + 0.02, -0.15])
    b1.visual.face_colors = box_color1
    scene.add_geometry(b1, node_name="box_1")
    
    b2 = trimesh.creation.box(extents=(0.32, 0.40, 0.32))
    b2.apply_translation([0.22, 0.15 + 0.20 + 0.02, 0.12])
    b2.visual.face_colors = box_color2
    scene.add_geometry(b2, node_name="box_2")

    # 2 boxes on middle shelf
    b3 = trimesh.creation.box(extents=(0.38, 0.30, 0.45))
    b3.apply_translation([0.15, 0.95 + 0.15 + 0.02, -0.1])
    b3.visual.face_colors = box_color1
    scene.add_geometry(b3, node_name="box_3")

    glb_data = scene.export(file_type="glb")
    with open(out_path, "wb") as f:
        f.write(glb_data)
    print(f"Exported {out_path} ({len(glb_data)} bytes)")


def create_robot_glb(out_path: str):
    scene = trimesh.Scene()
    
    # AMR chassis: 0.72m x 0.22m x 0.72m rounded box or beveled box
    chassis_color = [45, 48, 55, 255]      # Matte dark charcoal
    deck_color = [70, 75, 85, 255]         # Metal top plate
    wheel_color = [20, 20, 22, 255]        # Black rubber
    led_color = [120, 220, 255, 255]       # Cyan LED
    
    # Base chassis
    chassis = trimesh.creation.cylinder(radius=0.36, height=0.18, sections=24)
    chassis.apply_translation([0, 0.12, 0])
    chassis.visual.face_colors = chassis_color
    scene.add_geometry(chassis, node_name="chassis")
    
    # Top rotating load deck / turnplate
    deck = trimesh.creation.cylinder(radius=0.28, height=0.04, sections=24)
    deck.apply_translation([0, 0.23, 0])
    deck.visual.face_colors = deck_color
    scene.add_geometry(deck, node_name="turntable")
    
    # Status light ring / strip - named 'LED' as required by Prompt 2.2
    # Torus ring on top edge
    led_ring = trimesh.creation.torus(major_radius=0.33, minor_radius=0.018, major_sections=24, minor_sections=12)
    led_ring.apply_translation([0, 0.19, 0])
    led_ring.visual.face_colors = led_color
    
    # Material named LED
    pbr_mat = trimesh.visual.material.PBRMaterial(
        name="LED",
        baseColorFactor=[0.5, 0.85, 1.0, 1.0],
        emissiveFactor=[0.6, 0.9, 1.0],
        roughnessFactor=0.2,
        metallicFactor=0.1
    )
    led_ring.visual.material = pbr_mat
    scene.add_geometry(led_ring, node_name="LED")
    
    # Direction heading marker (small notch/beak at front +z)
    front_marker = trimesh.creation.box(extents=(0.06, 0.03, 0.10))
    front_marker.apply_translation([0, 0.20, 0.32])
    front_marker.visual.face_colors = [255, 200, 30, 255]
    scene.add_geometry(front_marker, node_name="heading_marker")
    
    # 2 drive wheels at sides (x = -0.34 and +0.34)
    for side in [-1, 1]:
        wheel = trimesh.creation.cylinder(radius=0.08, height=0.04, sections=16)
        # Rotate cylinder axis to lie along X
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        wheel.apply_transform(rot)
        wheel.apply_translation([side * 0.34, 0.08, 0])
        wheel.visual.face_colors = wheel_color
        scene.add_geometry(wheel, node_name=f"wheel_{side}")
        
    glb_data = scene.export(file_type="glb")
    with open(out_path, "wb") as f:
        f.write(glb_data)
    print(f"Exported {out_path} ({len(glb_data)} bytes)")


def create_barrier_glb(out_path: str):
    scene = trimesh.Scene()
    
    orange = [240, 80, 20, 255]
    white = [245, 245, 245, 255]
    black = [35, 35, 40, 255]
    
    # Safety traffic cone: square base + stepped cone with reflective stripe
    base = trimesh.creation.box(extents=(0.48, 0.04, 0.48))
    base.apply_translation([0, 0.02, 0])
    base.visual.face_colors = black
    scene.add_geometry(base, node_name="base")
    
    # Lower cone section (orange)
    cone_lower = trimesh.creation.cone(radius=0.20, height=0.30, sections=20)
    cone_lower.apply_translation([0, 0.17, 0])
    cone_lower.visual.face_colors = orange
    scene.add_geometry(cone_lower, node_name="cone_lower")
    
    # Reflective white middle stripe
    stripe = trimesh.creation.cylinder(radius=0.13, height=0.12, sections=20)
    stripe.apply_translation([0, 0.36, 0])
    stripe.visual.face_colors = white
    scene.add_geometry(stripe, node_name="cone_stripe")
    
    # Upper cone section (orange)
    cone_upper = trimesh.creation.cone(radius=0.09, height=0.24, sections=20)
    cone_upper.apply_translation([0, 0.49, 0])
    cone_upper.visual.face_colors = orange
    scene.add_geometry(cone_upper, node_name="cone_upper")
    
    glb_data = scene.export(file_type="glb")
    with open(out_path, "wb") as f:
        f.write(glb_data)
    print(f"Exported {out_path} ({len(glb_data)} bytes)")


def main():
    out_dir = "viewer/public/models"
    os.makedirs(out_dir, exist_ok=True)
    
    create_shelf_glb(os.path.join(out_dir, "shelf.glb"))
    create_robot_glb(os.path.join(out_dir, "robot.glb"))
    create_barrier_glb(os.path.join(out_dir, "barrier.glb"))
    print("All models generated successfully!")


if __name__ == "__main__":
    main()
