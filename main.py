import cv2

from src.aruco_calibrator import ArucoCalibrator
from src.camera import Camera
from src.config import WINDOW_INITIAL_HEIGHT, WINDOW_INITIAL_WIDTH, WINDOW_NAME
from src.debris_detector import DebrisDetector
from src.gesture_controller import GestureController
from src.measurer import MeasureStatus, Measurer
from src.renderer import Renderer
from src.snapshot import save_snapshot


def main():
    camera = Camera()
    calibrator = ArucoCalibrator()
    detector = DebrisDetector()
    measurer = Measurer()
    renderer = Renderer()
    gestures = GestureController()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT)

    print("Medidor de Detritos iniciado.")
    print("Controles: S=salvar | T=modo threshold | D=mascara debug | R=reset | Q=sair")
    print("Gesto: feche o punho e abra mostrando o indicador para salvar.")
    print("Dica: arraste a borda da janela para redimensionar.")

    show_debug_mask = False

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Falha na leitura da webcam.")
                break

            marker_corners, aruco_mask = calibrator.process(frame)
            calibrator.draw(frame, marker_corners)

            contour, binary = detector.detect(frame, aruco_mask)
            measurement = measurer.measure(contour, calibrator.px_per_cm)

            gesture_trigger, frame = gestures.process(frame)
            display = renderer.draw(
                frame,
                measurement,
                contour,
                detector.mode,
                gesture_active=gesture_trigger,
                debug_mask=binary if show_debug_mask else None,
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                calibrator.reset()
                measurer.reset()
                print("Suavização resetada.")
            if key in (ord("t"), ord("T")):
                mode = detector.cycle_mode()
                print(f"Modo threshold: {mode.value}")
            if key in (ord("d"), ord("D")):
                show_debug_mask = not show_debug_mask
                print(f"Máscara de debug: {'ligada' if show_debug_mask else 'desligada'}")
            if key in (ord("s"), ord("S")) or gesture_trigger:
                if measurement.status == MeasureStatus.OK:
                    save_snapshot(display, measurement, calibrator.px_per_cm)
                else:
                    print(f"Não é possível salvar: {measurement.status.value}")

    finally:
        gestures.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
