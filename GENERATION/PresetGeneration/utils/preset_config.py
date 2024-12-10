class PresetConfig:
    def __init__(self, class_name):
        self.config = dict()
        self.config["class_name"] = class_name

        if self.config["class_name"] == "mouth":
            self.config["num_shapes"] = 5
            self.config["prompts"] = [
                "with an open mouth",
                "with mouth wide open",
                "frowning",
                "with tongue out",
                "with vampire teeth",
            ]
            self.config["shape_ids"] = ["0", "1", "2", "3", "5"]

        if self.config["class_name"] == "mouth_talk":
            self.config["num_shapes"] = 2
            self.config["prompts"] = [
                "with a mouth with teeth",
                "with a mouth with lips",
                # "with a big open mouth",
                # "with a line-drawing of mouth",
                # "with a small round mouth",
                # "with a line-drawing of mouth",
                # "with a line-drawing of mouth",
                # "with a line-drawing of mouth",
                # "with a line-drawing of mouth"
            ]
            self.config["shape_ids"] = ["9", "10"] #, "8"] #, "1", "2", "3", "4", "5", "6"]
        
        if self.config["class_name"] == "arpa_mouth":
            self.config["num_shapes"] = 20
            self.config["prompts"] = ['with a mouth']*self.config["num_shapes"]
            self.config["shape_ids"] = [str(i) for i in range(self.config["num_shapes"])]
        
        if self.config["class_name"] == "arpabets":
            self.config["num_shapes"] = 14
            self.config["prompts"] = ['with a mouth']*self.config["num_shapes"]
            self.config["shape_ids"] = [str(i) for i in range(self.config["num_shapes"])]
            # self.config["key_ref_ids"] = [1, 2, 7, 9]
            # self.config["ref_mapping"] = {0:2, 1:-1, 2:-1, 3:1, 4:1, 5:1, 6:1, 7:-1, 8:7, 9:-1, 10:7, 11:2}
            # self.config["ref_mapping"] = {0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:-1, 8:-1, 9:-1, 10:-1, 11:-1}
            # self.config["generation_order"] = [1,2,7,9,0,3,4,5,6,8,9,10,11]

        # if self.config["class_name"] == "eyes":
        #     self.config["num_shapes"] = 12
        #     # self.config['prompts'] = ['with eyes and pupils', 'with eyes looking left',  'with eyes looking right', 'with small pupil', 'with big pupil', 'with big shiny eyes', 'with big shiny eyes', 'with eyes looking up', 'with eyes looking down', 'with eyes slightly closed', 'with happy eyes', 'with eyes closed']
        #     # self.config['shape_ids'] = ['0','1','2','3','4','5', '6','7','8','9','10','11']

        #     self.config["prompts"] = [
        #         "with dark pupils looking left",
        #         "with big shiny dark eyes",
        #         "with pupils looking up",
        #         "with eyes slightly closed",
        #         "with happy eyes",
        #         "with eyes closed",
        #     ]
        #     # self.config['shape_ids'] = ['1','5','7','9','10','11']
        #     self.config["shape_ids"] = ["5", "9", "10", "11"]

        if self.config["class_name"] == "eyes":
            self.config["num_shapes"] = 8
            self.config["prompts"] = ['with eyes and pupil',
                                      'with partially closed eyes',
                                      'with closed eyes',
                                      'with happy eyes',
                                      'with big dark shiny eyes',
                                      'with dark cross eyes',
                                      'with shocked eyes',
                                      'with eyes looking sideways'
                                      ]
            self.config["shape_ids"] = [str(i) for i in range(self.config["num_shapes"])]