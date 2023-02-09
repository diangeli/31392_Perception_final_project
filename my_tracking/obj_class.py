


class obj_on_conveyr:

    def __init__(self, BB, start_roi) -> None:
        self.BBx    = BB[0] + start_roi[1]      # The x of the BB
        self.BBy    = BB[1] + start_roi[0]      # The y of the BB
        self.w      = BB[2]                     # The width of the object
        self.h      = BB[3]                     # The height of the object
        self.area   = self.h * self.w           # The area of the object

        self.start_roi = start_roi              # The start region of interest

        self.type   = None                      # The type of object
        self.active = True                      # If the object is still on the conveyr
        

        self.calc_center()                      # [x, y]
        self.prev_pos()                         # Creating the previous pos variables



    def calc_center(self):
        x_center = int(self.BBx + self.h/2)
        y_center = int(self.BBy + self.w/2)

        self.center = [x_center, y_center]
        
    def prev_pos(self):
        self.prevx = self.center[0]
        self.prevy = self.center[1]

        self.prevRbound = self.BBx + self.w
        self.prevBbound = self.BBy + self.h

    def update_pos(self, BB):
        # # Update the prev pos, before overwritting the current pose
        # self.prev_pos()

        # Overwriting current pose
        self.BBx    = BB[0]                     # The x of the BB
        self.BBy    = BB[1]                     # The y of the BB
        self.w      = BB[2]                     # The width of the object
        self.h      = BB[3]                     # The height of the object

        self.calc_center()


    def kalman_init_params(self, K_params):
        self.x = K_params[0]
        self.P = K_params[1]
        self.u = K_params[2]
        self.F = K_params[3]
        self.H = K_params[4]
        self.R = K_params[5]


    












