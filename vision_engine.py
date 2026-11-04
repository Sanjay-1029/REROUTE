import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime


class VisionEngine:

    def __init__(self, model_path="yolo11n.pt", confidence_threshold=0.25):

        self.camera = None
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        print("REROUTE Vision Engine initialized.")
        print("YOLO object detection model loaded.")

    # IMAGE INPUT

    def load_image(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        print("Image loaded successfully.")
        return image

    # IMAGE PROCESSING

    def process_image(self, image):

        if image is None:
            raise ValueError("Invalid image input.")

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # EDGE-BASED OBSTACLE CANDIDATES

    def detect_obstacles(self, image):

        gray = self.process_image(image)

        edges = cv2.Canny(gray, 50, 150)

        return edges

    def detect_obstacle_regions(self, image):

        edges = self.detect_obstacles(image)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > 500:

                x, y, w, h = cv2.boundingRect(contour)

                regions.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": float(area)
                })

        return regions

    # YOLO OBJECT DETECTION

    def detect_objects(self, image):

        if image is None:
            raise ValueError("Invalid image input.")

        results = self.model(
            image,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []

        image_height, image_width = image.shape[:2]

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                position = self.classify_position(
                    x1,
                    x2,
                    image_width
                )

                detections.append({

                    "object": self.model.names[class_id],
                    "class_id": class_id,
                    "confidence": round(confidence, 3),
                    "position": position,

                    "bounding_box": [
                        x1, y1, x2, y2
                    ],

                    "image_width": image_width,
                    "image_height": image_height

                })

        return detections

    # IMAGE POSITION CLASSIFICATION

    def classify_position(self, x1, x2, image_width):

        object_center = (x1 + x2) / 2
        frame_center = image_width / 2

        if object_center < frame_center * 0.67:
            return "LEFT"

        elif object_center > frame_center * 1.33:
            return "RIGHT"

        return "CENTER"

    # UNCERTAINTY AND RISK ASSESSMENT

    def assess_obstacle_risk(self, detection):

        confidence = detection["confidence"]
        position = detection["position"]
        object_name = detection["object"]

        if confidence < 0.40:

            risk = "UNCERTAIN"
            uncertainty = "HIGH"

        elif position == "CENTER":

            risk = "POTENTIAL_HAZARD"
            uncertainty = "MEDIUM"

        else:

            risk = "NEEDS_INSPECTION"
            uncertainty = "MEDIUM"

        return {

            "object": object_name,
            "position": position,
            "confidence": confidence,
            "risk_level": risk,
            "uncertainty": uncertainty,
            "requires_verification": True

        }

    # BUILD VISUAL ENVIRONMENT STATE

    def build_environment_state(self, image):

        detections = self.detect_objects(image)

        observations = []

        for detection in detections:

            risk_assessment = self.assess_obstacle_risk(
                detection
            )

            observations.append({

                "object": detection["object"],
                "class_id": detection["class_id"],
                "confidence": detection["confidence"],
                "position": detection["position"],

                "bounding_box": detection["bounding_box"],

                "risk_level": risk_assessment["risk_level"],
                "uncertainty": risk_assessment["uncertainty"],

                "obstacle_candidate": True,
                "requires_verification": True

            })

        uncertain_count = sum(
            1 for item in observations
            if item["uncertainty"] == "HIGH"
        )

        return {

            "objects_detected": len(observations),

            "observations": observations,

            "uncertain_detections": uncertain_count,

            "perception_status": (
                "OBJECTS_DETECTED"
                if observations
                else "NO_OBJECTS_DETECTED"
            ),

            "timestamp": datetime.now().isoformat()

        }

    # NAVIGATION AND VISUAL REASONING INTEGRATION

    def prepare_reasoning_context(self, navigation_context, image):

        if not isinstance(navigation_context, dict):

            raise TypeError(
                "Navigation context must be a dictionary."
            )

        context = navigation_context.copy()

        visual_state = self.build_environment_state(image)

        context["visual_observations"] = (
            visual_state["observations"]
        )

        context["visual_detection_count"] = (
            visual_state["objects_detected"]
        )

        context["uncertain_detections"] = (
            visual_state["uncertain_detections"]
        )

        context["perception_source"] = "YOLO"

        context["perception_status"] = (
            visual_state["perception_status"]
        )

        context["perception_timestamp"] = (
            visual_state["timestamp"]
        )

        context["visual_navigation_warning"] = any(
            item["risk_level"] == "POTENTIAL_HAZARD"
            for item in visual_state["observations"]
        )

        context["perception_limitation"] = (
            "Image-relative positions do not represent "
            "physical distance or grid coordinates. "
            "Visual detections are candidates, not confirmed "
            "navigation obstacles."
        )

        return context

    # VISUALIZATION

    def display_results(self, edges):

        cv2.imshow(
            "REROUTE - Visual Perception",
            edges
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # IMAGE ANALYSIS

    def analyze_image(self, image_path):

        image = self.load_image(image_path)

        detections = self.detect_objects(image)

        print("\n========== REROUTE VISUAL ANALYSIS ==========")

        for detection in detections:

            risk = self.assess_obstacle_risk(detection)

            print(
                f"Object: {detection['object']} | "
                f"Confidence: {detection['confidence']} | "
                f"Position: {detection['position']} | "
                f"Risk: {risk['risk_level']}"
            )

            x1, y1, x2, y2 = detection["bounding_box"]

            label = (
                f"{detection['object']} "
                f"{detection['confidence']:.2f}"
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        print(
            f"\nTotal objects detected: {len(detections)}"
        )

        cv2.imshow(
            "REROUTE - YOLO Object Detection",
            image
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# STANDALONE TEST

if __name__ == "__main__":

    vision = VisionEngine()

    image_path = "test_image.jpg"

    vision.analyze_image(image_path)