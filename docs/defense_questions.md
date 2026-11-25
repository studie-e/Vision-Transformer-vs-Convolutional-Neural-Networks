# Câu hỏi vấn đáp và điểm cần giải thích

1. 10% là 4.500 ảnh vì mẫu số là train sau tách validation. Vì sao dùng tập con lồng nhau? Cách hai mô hình chia sẻ cùng ảnh trong từng cặp seed?
2. Split seed thay cả train và validation; train seed thay khởi tạo/shuffle. SD giữa các cặp bao gồm những nguồn biến thiên nào?
3. Top-1 accuracy của 5.000 ảnh thay đổi theo bước 1/5.000. Vì sao 0,2647 cho biết log cần kiểm tra lại?
4. Vì sao checkpoint tốt nhất dùng validation và hòa lấy epoch sớm hơn? Checkpoint cuối có mục đích gì?
5. So sánh 100 epoch ở ba mức dữ liệu có cùng số bước không? Tính ceil(N/128)×100.
6. ViT dùng patch 4×4 nên có 64 patch và CLS thành 65 token. Vì sao attention có shape [B,4,65,65]?
7. Trích CLS → patch block cuối trung bình head khác attention rollout thế nào? Heatmap 8×8 ánh xạ lên ảnh gốc ra sao?
8. Nếu bỏ positional embedding, số tham số giảm 65×96. Vì sao cần cặp seed/recipe giống baseline khi phân tích ablation?
9. Residual shortcut và projection trong ResNet làm gì? Vì sao logits đi trực tiếp vào CrossEntropy?
10. Vì sao ResNet mean ba seed và ViT seed đầu không đủ làm kết luận so sánh ba seed? Cần bổ sung bằng chứng nào?
11. Resume cần optimizer, scheduler và RNG nào? Vì sao chỉ đặt lại seed chưa đủ?
12. Có thể suy nguyên nhân từ một attention map hay từ chênh lệch accuracy của hai recipe không? Thiết kế thí nghiệm bổ sung thế nào?
13. Test cuối có được dùng để đổi LR/chọn seed không? Cách lưu registry và checkpoint hash giúp truy vết kết quả thế nào?

Diễn tập mỗi người 5–7 phút, ghi câu trả lời thực tế và câu còn chưa giải thích được sau buổi tập.
