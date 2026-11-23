# Kết quả thí nghiệm với các bộ tối ưu

[English](RESULTS.md) · [Số liệu dạng JSON](results.json) · [Hướng dẫn chạy lại](README.md)

Sau 120 epoch, SGD có động lượng cho α trung bình có trọng số của các lớp gần 2 trên cả MNIST và Fashion-MNIST. Khi giữ nguyên hướng cập nhật của AdamW nhưng dùng độ dài bước của SGD có động lượng, α trung bình tăng trong cả 3 phép so sánh. Trên MNIST và trong đợt huấn luyện tiếp của Fashion-MNIST, phép sửa này đưa α trung bình về gần 2. Với lịch huấn luyện Fashion-MNIST ban đầu, AdamW đã cho α trung bình lớn hơn 2 và phép sửa đẩy giá trị này lên cao hơn.

| Dữ liệu và lịch huấn luyện | Số seed ghép cặp | SGD có động lượng | AdamW | AdamW đã sửa |
|---|---:|---:|---:|---:|
| MNIST (chữ số) | 5 | 2,025 | 1,855 | 2,043 |
| Fashion-MNIST, lịch ban đầu | 3 | 1,992 | 2,188 | 2,646 |
| Fashion-MNIST, huấn luyện tiếp | 2 | 2,016 | 1,867 | 2,067 |

Mỗi ô là kết quả trung bình trên cùng các seed ở epoch 120, với batch 256 và tốc độ học danh định. Trong mỗi mô hình, trọng số của từng lớp phụ thuộc vào số trị riêng ở phần đuôi và độ khớp của phân bố lũy thừa. Trên MNIST, cả 5 seed đều cho α trung bình gần 2 hơn sau khi sửa AdamW.

**Độ chính xác và α.** Trên toàn bộ các lần chạy Fashion-MNIST, checkpoint AdamW tốt nhất đạt độ chính xác kiểm thử 90,35% tại α trung bình 2,296. Với AdamW đã sửa, kết quả là 90,26% tại 2,002. Hình 15 gộp các lịch huấn luyện và dùng màu để chỉ số epoch đã hoàn thành. Tam giác xanh đánh dấu checkpoint có độ chính xác cao nhất trong mỗi ô. Nếu bằng điểm, chọn checkpoint có α trung bình gần 2 nhất. [Hình 15](plots/figure15_accuracy.png)

**Độ chênh ERG được tính bằng số trị riêng.** Ở epoch 120, trong 254 lớp ẩn có α > 2, có 246 lớp có độ chênh dương. Trong 282 lớp có α < 2, có 133 lớp có độ chênh âm. Trên MNIST, lớp có α = 1,99953 cho độ chênh chỉ +1 trị riêng. Trên Fashion-MNIST, một lớp có α = 2,00069 vẫn có độ chênh +38. Cách chuẩn hóa có thể góp phần giải thích độ chênh còn lại. [Hình 16](plots/figure16_erg_gap.png)

**Với Muon, α tiếp tục thay đổi sau khi độ chính xác đã ổn định.** Từ epoch 200 đến 500, α trung bình không trọng số của FC1, FC2 và FC3 giảm từ 2,402 ± 0,339 xuống 2,124 ± 0,170. Độ chính xác kiểm thử chỉ thay đổi từ 99,23% xuống 99,22%. Dấu ± biểu thị độ lệch chuẩn mẫu giữa 3 lớp. Hình 17 là hình gốc của lần chạy dài. Hình 18 dùng kết quả của một thí nghiệm riêng với 3 seed và 30 epoch. [Hình 17](figure9_mnist_optimizer_comparison_panelgrid.png) · [Hình 18](plots/figure18_muon_30.png)

**Độ tin cậy của dự đoán.** Ở cả 75 lần chạy Fashion có độ chính xác ổn định và đủ dữ liệu, khoảng cách giữa xác suất dự đoán và tỷ lệ dự đoán đúng tăng trong nhóm dự đoán có độ tin cậy từ 99% trở lên. Trong 2 lần chạy Fashion AdamW không tăng cường dữ liệu, 10% ảnh khó nhất đóng góp 91,26% mức tăng của hàm mất mát log trên tập kiểm thử. Với mỗi lần chạy, nhóm ảnh khó được xác định bằng dự đoán ở epoch 60 của seed còn lại.

## Cách kiểm tra

Trong thư mục này, cài các gói bằng `python -m pip install -r requirements.txt`, rồi chạy `python reproduce.py` với Python 3.12. Mã sẽ đọc các trị riêng đã lưu, ước lượng lại α, kiểm tra số liệu và tạo 4 trang hình. Tệp `build/results.json` chứa kết quả tính lại. Hình 17 dùng lại ảnh gốc vì notebook của lần chạy dài chỉ lưu một số checkpoint dưới dạng số. Muốn huấn luyện lại từ đầu hoặc kiểm tra toàn bộ logits gốc, cần các tệp lưu trữ riêng. [Mô tả dữ liệu](INPUTS.md)
