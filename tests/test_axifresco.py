from math import cos, sin, pi
from axifresco.axifresco import Axifresco, Shape, Point, FORMATS


class TestAxifresco:
    @staticmethod
    def test_draw_shape_v2():
        ax = Axifresco()

        res = 64
        points = [Point(FORMATS.A3.x/2 + 20 * cos(i * 2 * pi / res), FORMATS.A3.y / 2 + 20 * sin(i * 2 * pi / res)) for i in range(res + 1)]
        s = Shape(points, is_polygonal=False)
        ax.draw_shape_v2(s)
        ax.move_home()


if __name__ == '__main__':
    TestAxifresco.test_draw_shape_v2()