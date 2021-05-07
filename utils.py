class ColorRange():
    def __init__(self, color: tuple, tnreshold=20, name='noname'):
        self.name = name
        self.color = color
        self.range_b = (max(self.color[0] - tnreshold, 0), min(self.color[0] + tnreshold, 255))
        self.range_g = (max(self.color[1] - tnreshold, 0), min(self.color[1] + tnreshold, 255))
        self.range_r = (max(self.color[2] - tnreshold, 0), min(self.color[2] + tnreshold, 255))

    def inrange(self, bgr_img):
        b, g, r = [bgr_img[:, :, i] for i in range(3)]
        remove_img_less = self.range_b[0] <= b
        remove_img_greater = self.range_b[1] >= b
        remove_imgb = remove_img_less * remove_img_greater
        remove_img_less = self.range_g[0] <= g
        remove_img_greater = self.range_g[1] >= g
        remove_imgg = remove_img_less * remove_img_greater
        remove_img_less = self.range_r[0] <= r
        remove_img_greater = self.range_r[1] >= r
        remove_imgr = remove_img_less * remove_img_greater
        thisarea = remove_imgr * remove_imgb * remove_imgg
        return thisarea

    def incolor(self, bgr_img):
        b, g, r = [bgr_img[:, :, i] for i in range(3)]
        remove_imgb = b == self.color[0]
        remove_imgg = g == self.color[1]
        remove_imgr = r == self.color[2]
        thisarea = remove_imgr * remove_imgb * remove_imgg
        return thisarea


eye_range = ColorRange((0, 0, 255), name='eye')
nose_range = ColorRange((0, 255, 0), name='nose')
mouth_range = ColorRange((255, 0, 0), name='mouth')
forehead_range = ColorRange((255, 0, 255), name='forehead')
cheek_range = ColorRange((0, 255, 255), name='cheek')
chin_range = ColorRange((255, 255, 0), name='chin')

range_list = [eye_range, nose_range, mouth_range, forehead_range, cheek_range, chin_range]