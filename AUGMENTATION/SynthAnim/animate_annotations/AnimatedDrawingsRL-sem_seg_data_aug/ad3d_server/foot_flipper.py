
import numpy as np
from typing import Tuple
import numpy.typing as npt


def get_edges(triangles) -> npt.NDArray[np.int32]:
    ret: npt.NDArray[np.int32] = np.empty([3*len(triangles), 2], dtype=np.int32)
    for idx, (v0, v1, v2) in enumerate(triangles):
        ret[3*idx+0, :] = [v0, v1]
        ret[3*idx+1, :] = [v1, v2]
        ret[3*idx+2, :] = [v2, v0]
    return ret


def get_intersected_edges(p1, p2, vertices, edges) -> Tuple[npt.NDArray[np.int32], int]:
    """
    Given p1 and p2, starting and ending points of line segment, along with vertices/edges of mask contour,
    determines what edges are crossed. Return the indices of those crossed edges and the distances from p1 to the crossings.
    Follows https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    """

    # create p and r
    p = np.full([edges.shape[0], 2], p1)
    r = np.full(p.shape, p2 - p1)  # shoot image left and see what edges we cross

    # create q and s
    q = vertices[edges[:, 0]]
    s = vertices[edges[:, 1]] - vertices[edges[:, 0]]

    # calculate t and u, use to check for edge intersection
    t = np.cross(q - p, s) / np.cross(r, s)
    u = np.cross(q - p, r) / np.cross(r, s)
    edge_idxs = np.squeeze(np.argwhere(np.logical_and(np.logical_and(0 <= t, t <= 1), np.logical_and(0 <= u, u <= 1))))
    edge_idxs = edge_idxs.reshape([-1, 1])

    return edge_idxs, t[edge_idxs]*np.linalg.norm(p2-p1)


def get_intersected_edges_though_boundary(edge_idxs, intersection_distances, edges, boundary_edges):

    edx_dist = list(zip(edge_idxs, intersection_distances))

    ret = []

    # starting with closest edge and progressing outward, check for boundary edge
    for edge_idx, dist in sorted(edx_dist, key=lambda x: x[1]):
        ret.append((dist, edge_idx))
        v0, v1 = np.squeeze(edges[edge_idx])
        if boundary_edges[tuple(sorted((v0, v1)))]:
            break

    # return list of all edges that were crossed on the way to boundary
    return ret


def get_edges_to_cut_and_new_vertices(cut_point, vertices, edges, boundary_edges):
    angles = list(range(10)) * np.array([np.pi/10])

    min_dist = 99999
    for angle in angles:
        v = 10000 * np.array([np.cos(angle), np.sin(angle)])

        # first half line
        edge_idxs1, intersection_distances1 = get_intersected_edges(cut_point, cut_point-v, vertices, edges)
        edx_dist1 = get_intersected_edges_though_boundary(edge_idxs1, intersection_distances1, edges, boundary_edges)
        d1, t1 = edx_dist1[-1]
        # second half line
        edge_idxs2, intersection_distances2 = get_intersected_edges(cut_point, cut_point+v, vertices, edges)
        edx_dist2 = get_intersected_edges_though_boundary(edge_idxs2, intersection_distances2, edges, boundary_edges)
        d2, t2 = edx_dist2[-1]

        if d1 + d2 < min_dist:
            min_dist = d1+d2
            edge1 = np.squeeze(t1)
            new_v1 = np.squeeze(cut_point - d1 * v/np.linalg.norm(v))

            edge2 = np.squeeze(t2)
            new_v2 = np.squeeze(cut_point + d2 * v/np.linalg.norm(v))

        return edge1, new_v1, edge2, new_v2


def foot_flip(char_cfg, outside_verts, segments, txtr_dict, prox_joint_name='left_knee', dist_joint_name='left_foot'):
    """
    Using a midpoint between prox and dist joint, finds a reasonble cut and mirrors the vertices on the distal end of that cut, about vector from prox->dist joint.
    Also modifies the txtrs in txtr_dict so by mirroring appropriate txtr pixels.
    """

    # get joint locations and the cut point midway between then
    prox_joint_x = np.array(char_cfg.skeleton[[x['name'] for x in char_cfg.skeleton].index(prox_joint_name)]['loc'])[0] * char_cfg.img_dim
    prox_joint_y = char_cfg.img_dim - np.array(char_cfg.skeleton[[x['name'] for x in char_cfg.skeleton].index(prox_joint_name)]['loc'])[1] * char_cfg.img_dim
    prox_joint_xy = np.array([prox_joint_x, prox_joint_y])

    dist_joint_x = np.array(char_cfg.skeleton[[x['name'] for x in char_cfg.skeleton].index(dist_joint_name)]['loc'])[0] * char_cfg.img_dim
    dist_joint_y = char_cfg.img_dim - np.array(char_cfg.skeleton[[x['name'] for x in char_cfg.skeleton].index(dist_joint_name)]['loc'])[1] * char_cfg.img_dim
    dist_joint_xy = np.array([dist_joint_x, dist_joint_y])

    cut_point = (prox_joint_xy + dist_joint_xy) / 2

    # # uncomment for debug visualization
    # from animated_drawings.utils import visualize_mesh
    # visualize_mesh(np.reshape(prox_joint_xy, [1, prox_joint_xy.shape[0]]), v_colors=[0, 0, 1], show_plot=False)
    # visualize_mesh(np.reshape(dist_joint_xy, [1, dist_joint_xy.shape[0]]), v_colors=[0, 0, 1], show_plot=False)
    # visualize_mesh(np.reshape(cut_point, [1, cut_point.shape[0]]), v_colors=[0, 1, 0], show_plot=False)
    # visualize_mesh(outside_verts)

    # get the segments to divide and the xy locations of the new vertices
    boundary_segments = {(v1, v2): True for v1, v2 in segments}
    e1, nv1, e2, nv2 = get_edges_to_cut_and_new_vertices(cut_point, outside_verts, segments, boundary_segments)

    ov_list = outside_verts.tolist()
    seg_list = segments.tolist()

    # # insert the new vertices into our list of vertices
    nv1_idx = len(ov_list)
    ov_list.append(nv1)
    nv2_idx = len(ov_list)
    ov_list.append(nv2)

    # create vector from prox_joint to dist_joint and use to test which segments lie within part to be flipped
    limb_v = dist_joint_xy - prox_joint_xy

    first_vert_to_flip = None
    vert_before_flipped = None
    last_vert_to_flip = None
    vert_after_flipped = None

    # remove the segments that have been cut, add new segments connecting their vertices to inserted vertices
    for seg, nv_idx in sorted([(e1, nv1_idx), (e2, nv2_idx)], key=lambda x: x[0], reverse=True):
        removed_seg = seg_list.pop(seg)
        first_seg = [removed_seg[0], nv_idx]
        second_seg = [nv_idx, removed_seg[1]]
        seg_list.append(first_seg)
        seg_list.append(second_seg)

        # check if the dot product between this new segment and the limb bone is positive. If so, this is the
        # first segment we use when determine what to flip.
        v_ = ov_list[removed_seg[1]] - ov_list[nv_idx]

        if np.isclose(0.0, np.linalg.norm(v_)):
            assert False, f'cut went exactly through boundary vertex, {prox_joint_name} to {dist_joint_name}'
        if 0 < np.dot(v_, limb_v) / (np.linalg.norm(v_) * np.linalg.norm(limb_v)):
            # if in direction of foot
            first_vert_to_flip = nv_idx
            vert_before_flipped = removed_seg[0]
        else:
            # if not in direction of foot
            last_vert_to_flip = nv_idx
            vert_after_flipped = removed_seg[1]

    assert first_vert_to_flip is not None
    assert vert_before_flipped is not None
    assert last_vert_to_flip is not None
    assert vert_after_flipped is not None

    # create dictionary to more easily traverse and modify edges
    seg_dict = {v0 : v1 for v0, v1 in seg_list}

    # get all the vertex idxs within limb we must flip
    verts_to_flip = []
    vert = first_vert_to_flip
    while True:
        verts_to_flip.append(vert)
        if vert == last_vert_to_flip:
            break
        vert = seg_dict[vert]  # follow segment countor to get next vertex

    # flip around this mid point
    xy = (ov_list[last_vert_to_flip] + ov_list[first_vert_to_flip]) / 2
    # and the direction will be vector perpendicular to the line
    vec = (ov_list[last_vert_to_flip] - ov_list[first_vert_to_flip])[::-1]
    vec = vec / np.linalg.norm(vec)

    # first we mirror the vertices, changing xy location
    for vert_idx in verts_to_flip:
        v_xy = ov_list[vert_idx]
        a =  v_xy - xy
        b = vec
        proj = b * np.dot(a, b) / np.dot(b, b)
        ov_list[vert_idx] = a + 2 * (proj-a) + xy

    # then we reverse the edges
    for idx in range(1, len(verts_to_flip)):
        seg_dict[verts_to_flip[idx]] = verts_to_flip[idx-1]

    # finally we connect it to the larger segment contour properly
    seg_dict[first_vert_to_flip] = vert_after_flipped
    seg_dict[vert_before_flipped] = last_vert_to_flip

    seg_list = [[k, v] for k, v in seg_dict.items()]

    # for each txtr, we need to rotate, mirror, the derotate.
    from PIL import Image, ImageOps
    import cv2
    for txtr in txtr_dict.values():
        # convert to PIL image with proper orientation
        img = Image.fromarray(np.flip(np.transpose(txtr, (1, 0, 2)), 0))

        # translate the image so that xy is in the very center
        cx, cy = img.width//2, img.height//2
        dx, dy = int(cx - xy[0]), int(cy - xy[1])
        img = img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, dy))

        # rotate so that vec points in -y axis
        theta = np.arctan2(vec[0], vec[1]) % np.pi
        img = img.rotate(theta)

        # mirror the image
        img = ImageOps.mirror(img)

        # rotate back
        img = img.rotate(-theta)

        # translate back
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, -dy))

        # copy flipped limb pixels onto original txtr
        img_arr = np.transpose(np.flip(np.array(img), 0), (1, 0, 2))

        # create a mask with the parts we want to keep
        mask = np.zeros([img_arr.shape[0], img_arr.shape[1], 3], np.uint8)
        mask_points = np.empty((1, len(verts_to_flip), 2), dtype=np.int32)
        for idx, vert_idx in enumerate(verts_to_flip):
            mask_points[0, idx] = [int(ov_list[vert_idx][1]), int(ov_list[vert_idx][0])]
        mask = cv2.fillPoly(mask, pts=mask_points, color=[255])[:, :, 0]

        txtr[mask == 255] = img_arr[mask == 255]

    return np.array(ov_list), np.array(seg_list)
