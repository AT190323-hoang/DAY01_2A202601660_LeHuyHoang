# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Ở temperature=0.0 và 0.7, model chọn ra những sự thật "an toàn", phổ biến
> (cà phê trứng, cầu Long Biên) và câu văn mạch lạc, đúng ngữ pháp hoàn
> toàn. Từ temperature=1.2 trở lên, model bắt đầu chọn từ ngữ ít khả năng
> hơn, cụ thể ở lần thử của tôi câu trả lời ở 1.2 bị chen một từ tiếng Nga
> ("знаменитый") ngay giữa câu tiếng Việt — dấu hiệu rõ ràng của việc mất
> mạch lạc. Ở 1.8 câu lại tình cờ mạch lạc trở lại (do bản chất ngẫu nhiên
> của sampling), nhưng nhìn chung xu hướng chung là: temperature càng cao,
> xác suất chọn phải token "lệch" càng lớn, và ngưỡng rủi ro rõ rệt bắt đầu
> quanh 1.0–1.2 trở lên.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Hợp đồng pháp lý: temperature thấp, khoảng 0.0–0.2. Văn bản pháp lý cần
> tính nhất quán, chính xác tuyệt đối về thuật ngữ và điều khoản — mọi độ
> "sáng tạo" đều là rủi ro (từ ngữ mơ hồ, sai điều khoản), nên cần model
> gần như luôn chọn token có xác suất cao nhất. Slogan quảng cáo: temperature
> cao, khoảng 1.0–1.3. Ở đây giá trị nằm ở sự bất ngờ, mới lạ, khác biệt so
> với slogan "an toàn" thường gặp — chấp nhận đánh đổi một phần mạch lạc để
> lấy đa dạng ý tưởng, vì người biên tập sẽ chọn lọc lại sau.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Tổng lượt gọi/ngày = 20.000 × 2 = 40.000 lượt. Tổng token đầu ra/ngày =
> 40.000 × 500 = 20.000.000 token = 20.000 nghìn-token. Với giá output
> trong bảng: GPT-4o (0.010 USD/1K) → 20.000 × 0.010 = **200 USD/ngày**;
> GPT-4o-mini (0.0006 USD/1K) → 20.000 × 0.0006 = **12 USD/ngày**. Chênh
> lệch ~16.7 lần (188 USD/ngày, ~68.600 USD/năm) — chỉ tính riêng chi phí
> output, chưa gồm input. Model lớn xứng đáng khi: tư vấn pháp lý/y tế,
> code review phức tạp — nơi 1 câu trả lời sai gây thiệt hại lớn hơn nhiều
> phần chênh lệch chi phí. Model nhỏ là lựa chọn đúng khi: chatbot FAQ,
> phân loại/tóm tắt đơn giản, autocomplete — khối lượng lớn, tác vụ không
> đòi hỏi suy luận sâu, chất lượng mini đã đủ dùng.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Gọi thật với cùng câu hỏi "Giải thích máy học là gì?". Persona nhà thơ
> trả lời hoàn toàn bằng ẩn dụ ("máy học như một chú chim nhỏ lần đầu tập
> hót", "mầm cây bé xíu dò dẫm vươn mình") — không một thuật ngữ kỹ thuật
> nào. Persona kỹ sư senior trả lời bằng liệt kê quy trình kỹ thuật thật
> (Data Collection → Data Cleaning → Model Training → Evaluation →
> Deployment/CI) và nêu ứng dụng cụ thể (nhận diện khuôn mặt, dịch thuật,
> gợi ý sản phẩm). Cùng một câu hỏi, cùng model, nhưng system prompt điều
> khiển được: giọng văn (thơ ca vs kỹ thuật), lựa chọn từ vựng (ẩn dụ vs
> thuật ngữ), và cấu trúc trình bày (văn xuôi liền mạch vs danh sách bước)
> — nội dung sự thật đằng sau vẫn là một, nhưng cách "đóng gói" hoàn toàn
> khác nhau.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Đoạn văn thử (177 từ, chủ đề AI) đo được: `count_tokens` (tiktoken) =
> **197 token**; ước lượng thô `177 / 0.75` = **236 token**. Chênh lệch
> ~19.8% — ước lượng thô cao hơn số thật. Nếu dùng ước lượng thô để dự
> toán ngân sách, ta sẽ **dự toán THỪA** (overestimate), vì hệ số 0.75
> (≈1.33 từ/token) được hiệu chỉnh dựa trên tiếng Anh — tiếng Việt có dấu
> thường được BPE tokenizer của OpenAI mã hóa hiệu quả hơn dự kiến (nhiều
> từ ghép/âm tiết gộp vừa 1 token), nên số token thật thấp hơn công thức
> thô tính ra. Dự toán thừa an toàn hơn thiếu (không bị vỡ ngân sách thực
> tế), nhưng nếu cần con số chính xác để định giá sản phẩm, nên dùng
> `count_tokens` thật thay vì ước lượng.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản (a) hưởng lợi nhiều nhất: người dùng nhìn thấy chữ xuất
> hiện ngay lập tức, cảm giác chờ đợi giảm hẳn dù tổng thời gian sinh ra
> câu trả lời không đổi — độ trễ "token đầu tiên" (TTFB) là yếu tố quyết
> định trải nghiệm ở đây. Trợ lý giọng nói đọc to (b) hưởng lợi ít hơn
> nhiều: TTS cần ít nhất trọn một câu (đôi khi cả đoạn) để phát âm đúng
> ngữ điệu, nên dù nhận được token ngay, hệ thống vẫn phải gom lại thành
> câu hoàn chỉnh trước khi đọc — lợi ích của streaming bị "hy sinh" một
> phần cho chất lượng giọng đọc tự nhiên. Pipeline dịch tài liệu chạy ngầm
> ban đêm (c) hoàn toàn không cần streaming: không có ai đang chờ xem kết
> quả theo thời gian thực, chỉ quan tâm bản dịch cuối cùng có đúng hay
> không khi job chạy xong, nên gọi non-streaming (một response trọn) vừa
> đơn giản hơn vừa không mất gì.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Với delay cố định, khi server quá tải và hàng nghìn client cùng gặp lỗi
> gần như đồng thời, tất cả sẽ retry lại đúng sau cùng một khoảng thời
> gian — tạo ra một đợt sóng request đồng loạt mới ("thundering herd") đập
> thẳng vào server đang cố hồi phục, khiến nó tiếp tục quá tải theo chu kỳ
> lặp lại. Exponential backoff giãn dần khoảng chờ (0.1s → 0.2s → 0.4s...)
> giúp giảm mật độ request dội vào cùng lúc, cho server thời gian hồi
> phục dần. Nhưng vấn đề chưa hết: vì mọi client bắt đầu thất bại gần như
> cùng một thời điểm, dù backoff tăng theo cấp số nhân, các client vẫn
> đồng bộ retry tại đúng cùng những mốc thời gian (tất cả cùng đợi đúng
> 0.4s, rồi cùng đợi đúng 0.8s...) — vẫn tạo ra các đợt sóng đồng loạt định
> kỳ, chỉ là thưa hơn. Jitter (cộng thêm một khoảng ngẫu nhiên nhỏ vào mỗi
> delay) phá vỡ tính đồng bộ đó, rải các lần retry ra ngẫu nhiên theo thời
> gian thay vì dồn cục tại cùng một mốc, giúp tải lên server mượt hơn thay
> vì theo từng đợt sóng.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt: "Bạn là trợ lý học tập cho sinh viên năm nhất ngành Khoa
> học Máy tính. **Luôn trả lời bằng tiếng Việt**, giải thích khái niệm kèm
> ví dụ code Python ngắn khi phù hợp. Nếu câu hỏi mơ hồ, hỏi lại để làm rõ
> trước khi trả lời. **Giữ câu trả lời dưới 150 từ trừ khi được yêu cầu chi
> tiết hơn.**"
>
> Hai chỗ quan trọng: (1) "Luôn trả lời bằng tiếng Việt" — nếu xóa, khi câu
> hỏi chứa nhiều thuật ngữ tiếng Anh (rất phổ biến trong CS), model có xu
> hướng trộn hoặc chuyển hẳn sang tiếng Anh, mất tính nhất quán ngôn ngữ
> cho người học mới. (2) "Giữ câu trả lời dưới 150 từ..." — nếu xóa, các
> câu trả lời có xu hướng dài dòng, liệt kê lan man; với giao diện CLI
> streaming, câu trả lời ngắn gọn đúng trọng tâm quan trọng hơn nhiều so
> với một bài giảng đầy đủ.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Tình huống: ở lượt 1, người dùng nói "Tôi bị dị ứng đậu phộng, đừng bao
> giờ gợi ý món có đậu phộng cho tôi." Sau đó cuộc trò chuyện tiếp tục 5-6
> lượt về các chủ đề nấu ăn khác (hỏi công thức phở, cách nêm nước mắm...).
> Đến lượt thứ 6-7, người dùng hỏi "Gợi ý cho tôi một món tráng miệng
> ngon" — vì history chỉ giữ 4 lượt cuối, thông tin dị ứng đậu phộng ở
> lượt 1 đã bị cắt khỏi context, trợ lý hoàn toàn có thể gợi ý một món có
> đậu phộng (ví dụ chè đậu phộng), gây hậu quả thực sự nguy hiểm chứ không
> chỉ là "trả lời sai" thông thường.
>
> Cách khắc phục: tách riêng "ràng buộc dài hạn" khỏi "history hội thoại
> ngắn hạn" — khi phát hiện người dùng nêu một thông tin mang tính lâu dài
> (dị ứng, sở thích cố định, ngữ cảnh nền), trích xuất và lưu vào một khối
> "ghi nhớ" cố định, luôn được nối vào ngay sau system prompt ở mọi lượt
> gọi, độc lập với phần history bị cắt theo cửa sổ trượt. Cách đơn giản
> hơn (ít tốn công hơn nhưng kém chính xác): định kỳ tóm tắt các lượt sắp
> bị cắt thành 1-2 câu, chèn vào đầu history thay vì xóa hẳn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
