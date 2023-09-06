# Vietnamese License Plate Recognition

Code báo cáo cuối khoá môn ML VinBigData của nhóm 12 với để tài nhận diện biển số xe trong hầm gửi xe

## Installation

```bash
  git clone https://github.com/Marsmallotr/License-Plate-Recognition.git
  cd License-Plate-Recognition

  # install dependencies using pip 
  pip install -r ./requirement.txt
```

- **Pretrained model** có trong folder model

## Run Seperatately

### License Plate Detection
```bash
  python lp_detection.py --img_path '{Tên đường dẫn ảnh đầu vào}' 
```

### License Plate Detection
- Input là ảnh xe trong hầm gửi xe 
- Output gồm các ảnh biển được crop và ảnh gốc được visualize lp detection
```bash
  python lp_detection.py --img_path '{Tên đường dẫn ảnh đầu vào}' --save_path '{Đường dẫn folder lưu kết quả}'
```


### License Plate Alignment
- Input là ảnh biển được crop ở bước trên 
- Output là ảnh biển đã được align lại
```bash
  python align_lp.py --img_path '{Tên đường dẫn ảnh biển}' --save_path '{Đường dẫn folder lưu kết quả}' --mode '{Phương pháp align (hough/keypoint)}'
```

### License Plate Recognition
- Input là ảnh biển được align ở bước trên 
- Output là string nội dung biển
```bash
  python lp_recognition.py --img_path '{Tên đường dẫn ảnh biển được align}'
```

## Run Whole Process
Chạy code trong file full_recognition.ipynb