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
    """
    Suppose each box is:

        [x1, y1, x2, y2]
         ↑   ↑   ↑   ↑
        left top right bottom
        
        Consider only the x-axis first.
        
        1. Left edge → max()
        
        Suppose:
        
        Box A: x1 = 2
        Box B: x1 = 5
        
        On a number line:
        
        2────────────10
             5────────────15
             ↑
           overlap starts
        
        The overlap cannot start at 2, because Box B doesn't exist before 5.
        
        So the overlap starts at the later/larger left boundary:
        
        intersection_x1 = max(2, 5)
                          # 5
        
        So:
        
        Left boundary of overlap = maximum of the two left boundaries.
        
        2. Right edge → min()
        
        Now suppose:
        
        Box A: x2 = 10
        Box B: x2 = 15
        2────────────10
             5──────────────15
                     ↑
                overlap ends
        
        The overlap cannot continue to 15, because Box A ends at 10.
        
        So the overlap ends at the earlier/smaller right boundary:
        
        intersection_x2 = min(10, 15)
                          # 10
        
        Therefore:
        
        Right boundary of overlap = minimum of the two right boundaries.
        
        3. The same logic applies to Y
        
        For the vertical direction:
        
        intersection_y1 = max(box_a[1], box_b[1])
        intersection_y2 = min(box_a[3], box_b[3])
        
        So the complete rule is:
        
                         INTERSECTION
        
                  ┌─────────────────────┐
                  │                     │
        LEFT  →   max                   │   ← RIGHT → min
                  │      OVERLAP         │
                  │                     │
                  └─────────────────────┘
                  ↑                     ↑
               TOP → max             BOTTOM → min
    """