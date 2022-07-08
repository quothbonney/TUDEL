class ApplicationState:
    def __init__(self):
        self.is_image_open = None
        self.original = None
        self.working_mask = [0]
        self.present = None
        self.type = None
        self.is_auto_masked = None
        self.is_masked = False
