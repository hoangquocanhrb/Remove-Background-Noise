# Remove-Background-Noise

**Là một phần của dự án robot hỗ trợ trong nhà hàng, mô hình có nhiệm vụ làm giảm tiếng ồn trong lời nói của khách hàng**

# Chuẩn bị dữ liệu

Dữ liệu gồm 2 phần: Giọng nói sạch (không có nhiễu) và nhiễu từ nhà hàng.
Giọng nói sạch được lấy từ [VinBigdata](https://vinbigdata.com/news/vinbigdata-chia-se-100-gio-du-lieu-tieng-noi-cho-cong-dong/) (100 giờ ghi âm), [VIVOS](https://ailab.hcmus.edu.vn/vivos) và file ghi âm thủ công. Nhiễu môi trường được thu bằng cách ghi âm tiếng ồn trong nhà hàng.

Tôi tập trung vào 3 loại nhiễu chính xuất hiện: Nhiễu do tiếng ồn bên trong nhà hàng, nhiễu do tiếng ồn xe cộ bên ngoài, nhiễu do dụng cụ bếp (đũa, thìa, bát, đĩa, xoong nồi).

Sau đó, trộn âm thanh nhiễu với giọng nói sạch với mức độ của âm thanh nhiễu ngẫu nhiên (20% đến 95%). Cuối cùng, dữ liệu huấn luyện gồm 6 giờ giọng nói có nhiễu (5 giờ training và 1 giờ validation).

Sử dụng mã nguồn mở của Vincent Belz[[1]https://github.com/vbelz/Speech-enhancement] để chuyển đổi tín hiệu âm thanh từ miền thời gian sang miền tần số

# Huấn luyện

Mô hình sử dụng mạng U-N với đầu vào là phổ biên độ sau khi chuyển qua miền tần số của giọng nói nhiễu, đầu ra là phần nhiễu của giọng nói.

<img src="images/denoise.gif">

*Source: Vincent Belz[[1](https://github.com/vbelz/Speech-enhancement)]*

Huấn luyện mô hình trên Google Colab

# Dự đoán
Mô hình sau khi huấn luyện có thể tải [tại đây](https://drive.google.com/file/d/1--3BAU2zYng-jtIP_4RLG-titOfNpThO/view?usp=sharing)
Lưu mô hình đã tải tại thư mục model

Ví dụ đầu vào tại dataset/Test/sound và kết quả tại dataset/Test/result_sound

# Cách sử dụng

- Clone repository này
- pip install -r requirements.txt
- Ghi âm giọng nói có nhiễu rồi lưu tại dataset/Test/sound, lưu ý trong thư mục sound chỉ có duy nhất 1 file
- python predict.py

Kết quả sẽ được lưu tại dataset/Test/result_sound

# Tham khảo

[1] Vincent Belz, "Speech-enhancement". Github:https://github.com/vbelz/Speech-enhancement

[2] Jansson, Andreas, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar and Tillman Weyde.Singing Voice Separation with Deep U-Net Convolutional Networks. ISMIR (2017).[https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf]

[3] Karol J. Piczak. 2015. ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd ACM international conference on Multimedia (MM '15). Association for Computing Machinery, New York, NY, USA, 1015–1018. DOI:[https://dl.acm.org/doi/10.1145/2733373.2806390]

