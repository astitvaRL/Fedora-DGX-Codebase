import numpy as np

########################  FACADES

# number_object = {
#                 0: 'background',###############
#                 1: 'facade',
#                 2: 'ledge',
#                 3: 'molding',
#                 4: 'pillar',
#                 5: 'deco', ###############
#                 6: 'cornice',
#                 7: 'window', #########
#                 8: 'sill',
#                 9: 'balcony', #########
#                 10: 'door',
#                 11: 'feb',
#                 12: 'fel',
#                 13: 'shop',   #########
#                 14: 'awning',
#                 15: 'sign',  #########
#                 16: 'tree', #########
#                 17: 'obs',  #########
# }
#
#
# number_color = {
#                 0: '#808080',
#                 1: '#751076',
#                 2: '#ff17ad'
#                 3: '#e8856e',
#                 4: '#ff4f03',
#                 5: '#854539',
#                 6: '#023e7f',
#                 7: '#05e8ff',
#                 8: '#027880',
#                 9: '#787bff',
#                 10: '#a56729',
#                 11: '#6729a5',
#                 12: '#a550a5',
#                 13: '#5dff73',
#                 14: '#fffe00',
#                 15: '#bde82a',
#                 16: '#3f6604',
#                 17: '#ff0000',
#
#
# }
#
#
# def color_pred(pred):
#
#     num_labels=18
#     color = np.array([[128,128,128],
#                     [117,16,118],
#                     [255,23,173],
#                     [232,133,110],
#                     [255,79,3],
#                     [133,69,57],
#                     [2,62,127],
#                     [5,232,255],
#                     [2,120,128],
#                     [120,123,255],
#                     [165,103,41],
#                     [103,41,165],
#                     [165,80,165],
#                     [93,255,115],
#                     [255,254,0],
#                     [189,232,42],
#                     [63,102,4],
#                     [255,0,0],
#                     ])
#     h, w = np.shape(pred)
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
#     #     print(color.shape)
#     for ii in range(num_labels):
#         #         print(ii)
#         mask = pred == ii
#         rgb[mask, None] = color[ii, :]
#     # Correct unk
#     unk = pred == 255
#     rgb[unk, None] = color[0, :]
#
#     return rgb


########################  FACES


# number_object = {
#                 0: 'background',
#                 1: 'skin',
#                 2: 'nose',
#                 3: 'eye_g',
#                 4: 'l_eye',
#                 5: 'r_eye',
#                 6: 'l_brow',
#                 7: 'r_brow',
#                 8: 'l_ear',
#                 9: 'r_ear',
#                 10: 'mouth',
#                 11: 'u_lip',
#                 12: 'l_lip',
#                 13: 'hair',
#                 14: 'hat',
#                 15: 'ear_r',
#                 16: 'neck_l',
#                 17: 'neck',
#                 18: 'cloth',
# }


# number_color = {
#                 0: '#000000',
#                 1: '#cc0000',
#                 2: '#4c9900',
#                 3: '#cccc00',
#                 4: '#3333ff',
#                 5: '#cc00cc',
#                 6: '#00ffff',
#                 7: '#33ffff',
#                 8: '#663300',
#                 9: '#ff0000',
#                 10: '#66cc00',
#                 11: '#ffff00',
#                 12: '#000099',
#                 13: '#0000cc',
#                 14: '#ff3399',
#                 15: '#00cccc',
#                 16: '#003300',
#                 17: '#ff9933',
#                 18: '#00cc00',

# }

#################### FACES DRAWINGS #################################
number_object = {
                0: 'BG',
                1: 'Eyebrows',
                2: 'Mouth',
                3: 'Pupil',
                4: 'Accessory',
                5: 'Skin',
                6: 'Nose',
                7: 'Teeth',
                8: 'Ear',
                9: 'Eye',
                10: 'Tongue',
}


number_color = {
                0: '#000000',
                1: '#4479e4',
                2: '#151c38',
                3: '#3f80f3',
                4: '#ab45ce',
                5: '#0c5730',
                6: '#a0eab3',
                7: '#791c4b',
                8: '#56a20b',
                9: '#f516a1',
                10: '#9a83c1',
}

#
# face_gray_color = np.array([[0,  0,  0],
#                     [204, 0,  0],
#                     [76, 153, 0],
#                     [204, 204, 0],##
#                     [51, 51, 255],##
#                     [204, 0, 204],##
#                     [0, 255, 255],##
#                     [51, 255, 255],##
#                     [102, 51, 0],##
#                     [255, 0, 0],##
#                     [102, 204, 0],##
#                     [255, 255, 0],##
#                     [0, 0, 153],##
#                     [0, 0, 204],##
#                     [255, 51, 153],##
#                     [0, 204, 204],##
#                     [0, 51, 0],##
#                     [255, 153, 51],
#                     [0, 204, 0],
#                     ])



# def color_pred(pred):

#     num_labels=19
#     color = np.array([[0,  0,  0],
#                     [204, 0,  0],
#                     [76, 153, 0],
#                     [204, 204, 0],##
#                     [51, 51, 255],##
#                     [204, 0, 204],##
#                     [0, 255, 255],##
#                     [51, 255, 255],##
#                     [102, 51, 0],##
#                     [255, 0, 0],##
#                     [102, 204, 0],##
#                     [255, 255, 0],##
#                     [0, 0, 153],##
#                     [0, 0, 204],##
#                     [255, 51, 153],##
#                     [0, 204, 204],##
#                     [0, 51, 0],##
#                     [255, 153, 51],
#                     [0, 204, 0],
#                     ])
#     h, w = np.shape(pred)
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
#     #     print(color.shape)
#     for ii in range(num_labels):
#         #         print(ii)
#         mask = pred == ii
#         rgb[mask, None] = color[ii, :]
#     # Correct unk
#     unk = pred == 255
#     rgb[unk, None] = color[0, :]

#     return rgb


def color_pred(pred):

    num_labels=11
    color = np.array([[0, 0, 0], [68, 121, 228], [21, 28, 56], [63, 128, 243], [171, 69, 206], [12, 87, 48], [160,234,179], [121,28,75], [86,162,11], [245,22,161], [154,131,193]])
    h, w = np.shape(pred)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)
    for ii in range(num_labels):
        #         print(ii)
        mask = pred == ii
        rgb[mask, None] = color[ii, :]
    # Correct unk
    unk = pred == 255
    rgb[unk, None] = color[0, :]

    return rgb
