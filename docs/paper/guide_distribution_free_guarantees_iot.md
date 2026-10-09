# **Hướng dẫn thực hiện bài AISTATS 2027 trên CIC IoT-DIAD 2024**

## **Mục tiêu và nguyên tắc**

Hai em hoàn thành thực nghiệm và bản paper đầy đủ trước **trưa 05/10**. Hạn nộp chính thức là 06/10 AoE, tức 18:59 ngày 07/10 giờ Việt Nam. Abstract đã đăng ký trên OpenReview.

Tên bài: **Distribution-Free Guarantees for IoT Intrusion Detection on Unseen Devices**.

### **Abstract (đã đăng ký trên OpenReview)**

> Intrusion detectors for Internet of Things networks are now trained on benchmarks rich enough that reported accuracy sits close to perfect, and small leaderboard gaps have become the main evidence of progress. Such single-number comparisons cannot tell whether an apparent gain reflects a real difference between methods. Nor can they tell how often a deployed model will miss a given attack family once it meets hardware it was never trained on. We study both questions on CIC IoT-DIAD 2024, a recent benchmark in which compromised IoT devices attack other devices inside a large testbed, with every capture described by packet-level and flow-level features. Our evaluation holds out entire devices rather than random rows, so each test prediction concerns traffic from a machine the model has never observed. On top of any trained classifier, we build class-conditional conformal prediction sets that bound the miss rate of each attack category without distributional assumptions. We then compare detectors and feature views through paired tests with multiple-comparison correction, together with bootstrap confidence intervals that resample whole devices. Under this protocol, several apparent improvements between strong detectors lose statistical support. The certified sets, by contrast, keep their promised coverage on rare categories, which is where point predictions degrade most. The width of each prediction set also reveals which traffic the packet and flow views fail to separate, and this gives analysts a principled signal for manual review. Because the guarantees rely only on exchangeability across held-out units, the protocol extends to other network benchmarks and to any detector that outputs scores.
>
> **Keywords:** intrusion detection, Internet of Things, conformal prediction, multiple comparisons, bootstrap inference, device-level generalization

Bài có bốn ý chính:

1. Đánh giá bằng cách giữ lại nguyên thiết bị, không chia ngẫu nhiên theo dòng.  
2. Dựng tập dự đoán conformal theo từng lớp để chặn tỉ lệ bỏ sót từng loại tấn công.  
3. So sánh mô hình bằng kiểm định cặp có hiệu chỉnh đa so sánh và bootstrap theo thiết bị.  
4. Dùng độ rộng tập dự đoán để chỉ ra lưu lượng khó tách.

Nguyên tắc làm việc:

* Trọng tâm là giao thức đánh giá và cam kết thống kê.  
* Không phê bình nhóm tác giả bộ dữ liệu.  
* Kết quả ra sao báo đúng như vậy. Nếu số liệu không ủng hộ kết luận trong abstract  
* Mọi lựa chọn ghi vào log\_quyet\_dinh.md, kèm lý do.  
* Cố định seed, lưu config, để Thầy chạy lại được.

## **Phân công**

SV A lo dữ liệu, mô hình và hình. SV B lo conformal, thống kê và lý thuyết. Hai em nối với nhau qua **file xác suất dự đoán** do SV A xuất ra.

| STT | Hạng mục | SV A (dữ liệu và mô hình) | SV B (conformal và thống kê) |
| :---- | :---- | :---- | :---- |
| 1 | Dữ liệu | Tải, thống kê, chống rò rỉ, viết data\_summary.md | Đọc data\_summary.md, kiểm tra số thiết bị mỗi lớp đủ cho calibration |
| 2 | Chia dữ liệu | Dựng S1, S2, S3 và chia train / calibration / test theo nhóm | Kiểm tra không thiết bị nào nằm ở hai phần |
| 3 | Mô hình | Huấn luyện 5 mô hình, xuất xác suất ra file | Viết code conformal chạy trên file xác suất |
| 4 | Thống kê | Tính macro-F1, recall từng lớp | Kiểm định cặp, Holm, BH, bootstrap theo thiết bị |
| 5 | Lý thuyết | Không | Mệnh đề, hệ quả, chứng minh trong phụ lục |
| 6 | Hình và bảng | F1, F2, T1, T2, T5 | F3, F4, F5, T3, T4 |
| 7 | Viết bài | Evaluation Protocol, Experiments, phụ lục thực nghiệm | Setting and Guarantees, Related Work, phụ lục chứng minh |
| 8 | Hoàn thiện | Gói code ẩn danh, README | Duyệt checklist viết, reproducibility checklist |

Introduction, Discussion và Conclusion: hai em viết nháp chung ngày 05/10, Thầy chỉnh.

## **Dữ liệu**

Bộ dữ liệu là **CIC IoT-DIAD 2024** của Canadian Institute for Cybersecurity, công bố trong bài Rabbani et al., IEEE Internet of Things Journal, 2024\. Bài báo phải trích dẫn công trình này.

Bộ dữ liệu có hai thư mục:

* **DI\_AD\_Packet-based-features**: đặc trưng theo từng gói tin, có cột device\_mac và các đặc trưng cửa sổ 1, 5, 10, 30, 60\.  
* **AD\_Flow-based-features**: đặc trưng theo luồng, trích bằng CICFlowMeter. Thư mục này có Src IP nhưng không có MAC.

Nhãn chính gồm 8 lớp: Benign, DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, Mirai. Nếu còn thời gian, dùng thêm nhãn chi tiết từng loại tấn công.

### **Việc đầu tiên (SV A)**

Thống kê cho từng thư mục rồi ghi vào data\_summary.md:

* Số dòng, số cột.  
* Phân bố lớp.  
* Số thiết bị và số file capture, kèm phân bố lớp theo từng thiết bị.

Lý do chọn bộ này, cần ghi vào bài: bộ còn mới; phục vụ cả nhận dạng thiết bị lẫn phát hiện bất thường; tấn công xuất phát từ chính thiết bị IoT, nên đặt được câu hỏi mô hình có tổng quát hoá sang thiết bị chưa thấy hay không.

### **Chống rò rỉ**

* Bỏ khỏi đặc trưng: Flow ID, IP, MAC, port định danh, Timestamp, stream id. Các cột này chỉ giữ lại để chia nhóm.  
* Xoá dòng trùng tuyệt đối. Báo tỉ lệ dòng trùng giữa train và test.  
* Nếu dữ liệu quá lớn so với máy: lấy mẫu phân tầng theo cặp (thiết bị, lớp), seed cố định, ghi rõ tỉ lệ lấy mẫu.