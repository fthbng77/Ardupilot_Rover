import rospy
import cv2
import numpy as np
import torch
from model import SegNet  # Varsayılan model olarak SegNet'i kullandım
from torchvision import transforms
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Parametreler
MODEL_PATH = "segnet_best_model.pth"  # Modelin kaydedildiği dosya yolu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli yükle
model = SegNet(input_channels=3, num_classes=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Dönüşüm işlevini tanımla
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

bridge = CvBridge()

CLASS_MAP = [
    (0, 'trail', (170, 170, 170)),
    (1, 'grass', (0, 255, 0)),
    (2, 'vegetation', (102, 102, 51)),
    (3, 'obstacle', (0, 0, 0)),
    (4, 'sky', (0, 120, 255)),
    (5, 'void', (0, 60, 0))
]

def mask_to_color(mask, original_shape):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(len(CLASS_MAP)):
        colored_mask[mask == i] = CLASS_MAP[i][2]
    
    # Reshape the heatmap to match the original shape
    colored_mask = cv2.resize(colored_mask, (original_shape[1], original_shape[0]))
    return colored_mask

def callback(data):
    try:
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # Görüntüyü tensora dönüştür
    tensor = transform(frame).unsqueeze(0).to(DEVICE)

    # Maskeyi tahmin et
    with torch.no_grad():
        prediction = model(tensor)
        mask = torch.argmax(prediction, dim=1).cpu().numpy()[0]

    # Maskeyi özelleştirilmiş renk haritasıyla renklendir
    heatmap = mask_to_color(mask, frame.shape)

    overlayed = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    # Görüntüyü göster
    cv2.imshow('Segmentation', overlayed)

    # 'q' tuşuna basarak pencereyi kapat
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('User Exit')

def main():
    rospy.init_node('image_segmentation', anonymous=True)
    rospy.Subscriber("/webcam/image_raw", Image, callback)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

