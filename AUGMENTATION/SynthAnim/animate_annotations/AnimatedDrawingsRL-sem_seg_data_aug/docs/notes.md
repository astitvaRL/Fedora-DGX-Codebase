## Adding back in depth-ordered rendering for body parts.

The Retargeter has a function, `get_current_frame_source_joint_depths()`, the is responsible for returning a dictionary with key-value pairs mapping motion source joint names to somelike like their z ordering.

AnimatedDrawingMesh knows about the mesh and is responsible for rendering it, so it's render order is what must be
modified. 
Therefore, AnimatedDrawingMesh needs to have some data structure mapping the name of the character joint to the group of triangles that belong to it. Futhermore, the order of triangle rendering within that group is also important and must not change.

But AnimatedDrawingMesh doesn't have knowledge about the rig or the joint names, specifically. Only AnimatedDrawingRig does.


During initialization, we will have AnimatedDrawingMesh_cardboard object perform the necessary steps to assign each triangle within its mesh to a particular bone, and will caches the indices related to each bone in the proper order.

- [x] pass rig into the initialization function

- [ ] AnimatedDrawingMesh.process_motion_source_depths(motion_source depths)
    - Abstract method, children should subclass. Any per-frame processing that requires the relative depths of the motion_source joints should be done here.

- [ ] AnimatedDrawingMesh_cardboard.rig_preprocessing(rig)
    - Creates the rendering order

- [ ] AnimatedDrawingMesh_cardboard.set_joint_rendering_order()

During run loop,
We rework `retargeter.get_current_frame_source_joint_depths()`, and instead create `retargeter.get_current_frame_character_joint_render_order`, which will return a list
of the character joints, in the order in which they should be rendered.
AnimatedDrawing object will call the retargeter, get this list, and pass it along to AnimatedDrawingMesh function `set_triangle_rendering_order_from_joint_list()`
So we have the AnimatedDrawing object call 
