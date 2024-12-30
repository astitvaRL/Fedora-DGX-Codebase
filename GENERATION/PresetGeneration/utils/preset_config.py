class PresetConfig:
    def __init__(self, class_name):
        self.config = dict()
        self.config["class_name"] = class_name

        
        if self.config["class_name"] == "mouth":
            self.config["num_shapes"] = 20
            self.config["prompts"] = ['with a mouth']*self.config["num_shapes"]
            self.config["shape_ids"] = [str(i) for i in range(self.config["num_shapes"])]


        if self.config["class_name"] == "eyes":
            self.config["num_shapes"] = 9
            self.config["prompts"] = ['with eyes and pupil',
                                      'with partially closed eyes',
                                      'with closed eyes',
                                      'with happy eyes',
                                      'with big dark shiny eyes',
                                      'with dark cross eyes',
                                      'with shocked eyes',
                                      'with eyes looking sideways'
                                      'with heart eyes'
                                      ]
            self.config["shape_ids"] = [str(i) for i in range(self.config["num_shapes"])]