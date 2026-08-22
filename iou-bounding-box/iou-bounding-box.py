import numpy as np

def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """

    """
    IoU is a metric used to measure how much two regions overlap.

    It is especially common in computer vision, such as:

        Object detection
        Image segmentation
        Bounding boxes
        Evaluating YOLO-style models

    The easiest way to understand it is with two boxes.

    Union

        Everything covered by either box, but don't count the
        overlapping region twice.

        Think:

        Union = Area(A) + Area(B) - Intersection

        Why subtract?

        Because the intersection was counted once in Area A
        and again in Area B.
    """

    box_a = np.asarray(box_a, dtype=float)
    box_b = np.asarray(box_b, dtype=float)

    # area a
    def areas(box):
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        return area

    area_a = areas(box_a)
    area_b = areas(box_b)

    # now we have to calculate intersection
    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])

    # if boxes do not overlap, width or height should be 0
    width = max(0, intersection_x2 - intersection_x1)
    height = max(0, intersection_y2 - intersection_y1)

    # area of intersection
    area_of_intersection = width * height

    # now union
    area_of_union = area_a + area_b - area_of_intersection

    # IoU
    iou_value = area_of_intersection / area_of_union

    return iou_value