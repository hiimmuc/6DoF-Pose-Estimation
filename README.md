# 6 DoF Object pose estimation

Investigate following method:

1. Simple way:

-   [x] Traditional: Segment (Image processing only) --> Get contour --> Extract Rotated Boxes
-   [x] Segment (DL: SAM2, Yolo) --> Get contour --> Extract Rotated Boxes --> Pose calculation with PnP
-   [x] Segment (DL: SAM2, Yolo) --> Get BBox (Yolo) --> Pose calculation with PnP

2. [ ] RGB-only: PoET
3. [ ] RGB-D: Any6D
4. Prioritize real-time inference:

-   [ ] RNNPose
-   [ ] Yolo v5 6D
-   [ ] GDRNPP (or Fast version)
