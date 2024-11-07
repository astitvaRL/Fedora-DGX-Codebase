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
            self.config["num_shapes"] = 3
            self.config["prompts"] = [
                "with a big open mouth",
                "with a line-drawing of mouth",
                "with a small round mouth",
            ]
            self.config["shape_ids"] = ["0", "1", "2"]

        if self.config["class_name"] == "eyes":
            self.config["num_shapes"] = 12
            # self.config['prompts'] = ['with eyes and pupils', 'with eyes looking left',  'with eyes looking right', 'with small pupil', 'with big pupil', 'with big shiny eyes', 'with big shiny eyes', 'with eyes looking up', 'with eyes looking down', 'with eyes slightly closed', 'with happy eyes', 'with eyes closed']
            # self.config['shape_ids'] = ['0','1','2','3','4','5', '6','7','8','9','10','11']

            self.config["prompts"] = [
                "with dark pupils looking left",
                "with big shiny dark eyes",
                "with pupils looking up",
                "with eyes slightly closed",
                "with happy eyes",
                "with eyes closed",
            ]
            # self.config['shape_ids'] = ['1','5','7','9','10','11']
            self.config["shape_ids"] = ["5", "9", "10", "11"]
