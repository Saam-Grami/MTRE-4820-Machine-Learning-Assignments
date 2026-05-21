# Ambulance Detection and Navigation Using CNN-Based Velocity Control
**MTRE 4820 — Machine Learning | Final Project**

---

## Data Preparation

Training data was collected inside a Gazebo simulation environment using a ROSbot equipped with a forward-facing RGB camera. The ambulance model was placed at 200 distinct positions and orientations within the robot's 60° field of view. For each configuration, the robot's pose and the ambulance's world coordinates were recorded alongside the corresponding camera image, which was saved and numbered sequentially. The ambulance was deliberately varied in both distance and lateral offset to ensure diversity across the dataset.

---

## Velocity Command Calculation

With a fixed linear velocity of 0.5 m/s, the only command that needs to be computed is the angular velocity ω. This was derived geometrically using a three-point circular arc connecting the robot (P1) and the ambulance (P2), with a third point (P3) controlling the curvature of the path.

**Finding θ:** The ambulance's horizontal position in the camera frame is captured as a pixel coordinate x. The angle θ between the robot's forward heading and the direction to the ambulance is recovered using the pinhole camera model: θ = arctan((x_pixel − cx) / fx), where cx = 320.5 px is the image center and fx = 554.38 px is the focal length in pixel units. The focal length acts as a synthetic depth, providing the triangle geometry needed to recover the angle even though the camera projects the 3D world onto a 2D plane. The 60° FOV naturally constrains θ to ±30°.

**Arc Construction:** The chord midpoint M is computed halfway between P1 and P2. A perpendicular offset of magnitude (L/4)·|sin θ| is applied at M to produce P3, where L is the chord length. The sin θ factor ensures the offset scales to zero when the ambulance is directly ahead, producing a straight path rather than an unnecessary curve. The sign of the offset is determined by the sign of θ, which encodes whether the ambulance is left or right of center.

**Radius and ω:** Three points uniquely define a circle. The center C is found by intersecting the perpendicular bisectors of segments P1P2 and P1P3. The radius is then R = |P1 − C|, and the angular velocity follows from ω = v / R. In ROS2, positive ω turns the robot left and negative ω turns it right.

---

## Model

A convolutional neural network was trained to predict angular velocity directly from camera images. The model used MobileNetV2 as a pretrained base with its weights frozen during an initial training phase, followed by a fine-tuning phase where the full network was unfrozen at a lower learning rate. Since only 200 images were available, the dataset was tripled to 600 samples by generating brightness and contrast augmented versions of each image, with labels unchanged since these transformations do not affect the ambulance's position in the frame. The output layer is a single linear node producing ω directly. While the model captured the general trend of the angular velocity well, predictions at the higher magnitude values were less precise — likely a consequence of the limited dataset size, which was constrained by the time required to collect samples through the Gazebo simulation environment.
A 70–15–15 split was chosen because it provides a good balance between training data and reliable evaluation while keeping the validation and test sets large enough for reliable results. This was chosen instead of an 80–10–10 split where the validation and test sets may be too small.

---

## ROS2 Integration

Within ROS2, a Python node subscribes to the ROSbot's camera topic, runs inference through the trained model on each incoming frame, and publishes the predicted angular velocity alongside the fixed linear velocity as a Twist message to /cmd_vel. The ambulance is spawned at a random position within the robot's field of view at the start of each run. As demonstrated in the accompanying video, the robot initially curves toward the ambulance as it appears to one side of the frame, and straightens its approach as the ambulance centers in the camera view. The node updates continuously at the rate camera frames arrive, meaning the correction frequency is ultimately limited by the hardware it runs on.
