# mqtt_subscriber.py
import paho.mqtt.client as mqtt
import cv2
import numpy as np
import oss2
from datetime import datetime
from detect_goose import run as detect_goose

# 巴法云 MQTT 配置
# MQTT_SERVER = "bemfa.com"
# MQTT_PORT = 1883
# MQTT_UID = "Your_Bemfa_UID"
# MQTT_KEY = "Your_Bemfa_SecretKey"
# TOPIC = "esp32cam/capture"

# # 阿里云 OSS 配置
# OSS_ACCESS_KEY = "Your_OSS_AccessKey"
# OSS_SECRET_KEY = "Your_OSS_SecretKey"
# OSS_ENDPOINT = "oss-cn-hangzhou.aliyuncs.com"
# OSS_BUCKET = "your-bucket-name"
#
# # 初始化 OSS 客户端
# auth = oss2.Auth(OSS_ACCESS_KEY, OSS_SECRET_KEY)
# bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)

# def on_connect(client, userdata, flags, rc):
#     print("Connected to MQTT")
#     client.subscribe(TOPIC)

def check():
    # # 接收图片并保存
    # img_data = np.frombuffer(msg.payload, dtype=np.uint8)
    # img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # input_path = f"temp/input_{timestamp}.jpg"
    # cv2.imwrite(input_path, img)
    input_path = "gooseANDegg/微信图片_20250329180726.jpg"

    # 调用 YOLOv5 检测
    output_path = f"temp/output_{timestamp}.jpg"
    result = detect_goose(
        weights="runs/train/exp16/weights/best.pt",
        source=input_path,
        project="temp",
        name="output",
        save_txt=True,
        save_conf=True
    )

    print (f'{result}')


    # 上传标注后的图片到 OSS
    # oss_object_name = f"annotated_{timestamp}.jpg"
    # bucket.put_object_from_file(oss_object_name, output_path)
    # oss_url = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{oss_object_name}"

    # 返回结果给微信小程序（需自行实现 HTTP API）
    # print(f"Count: {count}, OSS URL: {output_path}")


# if __name__ == "__main__":
#     client = mqtt.Client()
#     # client.username_pw_set(MQTT_UID, MQTT_KEY)
#     # client.on_connect = on_connect
#     client.on_message = on_message
#     # client.connect(MQTT_SERVER, MQTT_PORT, 60)
#     client.loop_forever()


if __name__ == "__main__":
    check()