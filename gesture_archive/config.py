from omegaconf import OmegaConf
                                                                                                                                                                                                                                                                                     

def generate_new_config():
    cfg = OmegaConf.create(
        {
            "ring_group": [
                # "FA56397A-A437-91EC-74A1-03F05C004962",
                # "D1285A7A-0E29-BD5F-914B-C59635D97CBF" # 906A
                # "8777611E-6E73-E81B-71E6-9B5F515D2263", # 7C1A
                # "A40C140A-A257-667C-98D8-48B12F3BFC85"
                # "6C10242E-1A24-9828-BA28-96B1E8A650FC"
                # "A40C140A-A257-667C-98D8-48B12F3BFC85"
                # "4E5F34CA-A4AA-30D8-0D34-A8BB6DCF1E00"
                # "3A639535-E8F2-0DCE-05B5-F45FEEB3B124" # 5087
                # "00D57A81-6BBF-0D04-B6BF-B74346BE4D39" # 4CA5
                # "4E5F34CA-A4AA-30D8-0D34-A8BB6DCF1E00"
                # "7CC1085C-0FA0-8E9C-6428-32F5B3B45368"
                # "A40C140A-A257-667C-98D8-48B12F3BFC85"
                # "308A3977-092E-7685-8A61-5C0CBC94C471"
                # "308A3977-092E-7685-8A61-5C0CBC94C471"
                # "89F1766B-5695-1B10-6857-49046A5A64D3" # 8393
                # "D369384D-D195-B1AD-690B-99B52511576C"
                # "42A97E8D-1508-6F1C-8FC9-4329DED0B921"
                # "F095AC1F-1BBD-4B06-AAEE-46A872BFAA3B"
                "ED4CF8AC-2C0C-DE71-FD0A-C75C0BB74FE0"
                # "6C10242E-1A24-9828-BA28-96B1E8A650FC"
                # "D369384D-D195-B1AD-690B-99B52511576C"
            ],
            "server_ip": "127.0.0.1",
            "server_port": 9998,
            "ring_app_data_buffer_max_len": 1000,
            "enable_gesture": True,
            "enable_imu_fps": True,
            "enable_gesture_fps": True,
            "allowed_gestures": [6, 8, 13, 14, 16, 36],
            # "allowed_gestures": [16, 25, 26, 30, 31, 34, 35, 36],
            # "allowed_gestures": [8, 13, 14, 16, 20, 21, 24, 25],
            "skipped_gestures": [36],
            "positive_ratio": 0.8,
            "result_window_length": 8,
            "window_size": 200,
            "models": [
                "checkpoint/it-pinch.pth",
                "checkpoint/it-snap.pth",
                # "checkpoint/gesture_cnn.pt",
            ],
            "models_allowed_gestures": [
                [8, 13, 14],
                [16, 30, 31, 34, 35, 36],
                # [8, 13, 14, 16, 20, 21, 24, 25]
            ],
            "device": "cpu",
            "selected_index": 0,
        }
    )

    return cfg

cfg = generate_new_config()

def save_config(cfg, path):
    OmegaConf.save(cfg, path)

if __name__ == "__main__":
    cfg = generate_new_config()
    save_config(cfg, "config.yaml")