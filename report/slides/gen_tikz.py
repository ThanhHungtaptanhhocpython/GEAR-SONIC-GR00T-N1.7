"""Sinh bộ slide hình của báo cáo Nvidia_groot.tex: các hình TikZ và các biểu đồ thống kê Bones-SEED.

Hình được tách từ báo cáo thành SVG trong assets/tikz/ (xem README.md, mục "Bộ slide hình TikZ").
Biểu đồ thống kê lấy thẳng từ report/figures/bones_seed/*.png (đường dẫn /_fig/...).
Script này ghi deck_tikz/deck.json và deck_tikz/slides/*.html theo mẫu trình bày của bộ slide chính.

Dùng:
    python report/slides/gen_tikz.py
    python report/slides/build_pdf.py --deck deck_tikz --out Slides_Hinh_TikZ_GR00T_SONIC.pdf
"""
import json
import pathlib
import re
import struct

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "deck_tikz"
TIKZ = HERE / "assets" / "tikz"
FIG = HERE.parent / "figures"

LOGO = "/_blob/199565b59f8ae0c9c6658b583fbd6750"
LOGO_BIG = "/_blob/f1ac4e10cfa6c9c1fb5c0e9c0bfcf6b5"
FONT = "font-family:'Be Vietnam Pro', Arial, sans-serif"

CHAPTERS = {
    1: "Tổng quan GR00T N1.7 và GEAR-SONIC",
    2: "Bộ dữ liệu Bones-SEED",
    "A": "Bộ dữ liệu Bones-SEED · Phụ lục A",
    3: "Tiền xử lý dữ liệu huấn luyện SONIC",
    4: "Huấn luyện GEAR-SONIC",
    5: "Ghép nối GR00T N1.7 với GEAR-SONIC",
}

# (tệp svg, số hình, tiêu đề, phụ đề, các ý chính cho bố cục dọc hoặc None, ghi chú thuyết trình)
FIGS = [
    ("gr00t_arch", "1.1", "Kiến trúc GR00T N1.7",
     "System 2 (VLM) hiểu ảnh và câu lệnh; System 1 (action head DiT) sinh chunk 40 bước hành động",
     None,
     "Khối VLM Cosmos-Reason2-2B, kiến trúc Qwen3-VL, đóng băng khi fine-tune: ảnh chia thành ô 32 × 32 điểm ảnh, "
     "ảnh 384 × 256 của nhóm cho 96 token; câu lệnh mẫu cho 13 token chữ, cộng 7 token định dạng, tổng 116 token; GR00T chỉ giữ 12 lớp đầu của mô hình ngôn ngữ, mỗi token 2048 chiều. "
     "Action head khử nhiễu 4 lần: trạng thái thành 1 token, 40 bước hành động thành 40 token, qua 16 khối DiT xen kẽ "
     "tự chú ý và chú ý chéo tới token chữ, token ảnh; bộ giải mã đổi mỗi token thành một bước, 78 số mỗi bước. "
     "Các khối màu xanh lá có trọng số riêng cho từng embodiment và được fine-tune."),
    ("sonic_3layer", "1.2", "Kiến trúc ba lớp của SONIC",
     "GR00T thay chỗ các encoder và xuất trực tiếp motion token cho decoder",
     ["<b>Lớp 1</b>: các nguồn sinh chuyển động, gồm Kinematic Planner, PICO VR, GEM và GR00T N1.7.",
      "<b>Lớp 2</b>: encoder và FSQ nén chuyển động thành motion token 64 chiều, 50 Hz.",
      "<b>Lớp 3</b>: decoder g1_dyn ghép token với proprioception, xuất góc khớp đích; bộ PD trong driver chạy 500 Hz.",
      "Dấu ngoặc trái: phần bài báo gọi là Universal Control Policy."],
     "Vẽ lại theo bài báo và mã nguồn. Decoder g1_kin (nét đứt) chỉ dùng khi huấn luyện. "
     "Universal Control Policy gồm ba encoder, FSQ và hai decoder; nguồn ở Lớp 1 và bộ PD không thuộc phần này."),
    ("gr00t_sonic_arch", "1.3", "Kiến trúc khi ghép GR00T với SONIC",
     "GR00T dùng đủ hai khối; SONIC chỉ dùng decoder g1_dyn",
     ["<b>GR00T thay chỗ encoder của SONIC</b>: motion token đi thẳng vào decoder g1_dyn, encoder và FSQ được bỏ qua.",
      "<b>Hai nhịp khác nhau</b>: GR00T trả 40 bước tối đa 2,5 lần/giây; client gửi 1 bước mỗi 20 ms; decoder chạy 50 Hz.",
      "<b>Phân chia trách nhiệm</b>: GR00T quyết định làm gì; SONIC điều khiển 29 khớp thân để làm được mà không ngã.",
      "Góc ngón tay đi thẳng từ GR00T tới bàn tay Dex3."],
     "Token GR00T xuất ra phải nằm trong cùng không gian token mà decoder đã học, nên nhãn token phải tạo bằng chính "
     "checkpoint SONIC sẽ dùng. Luồng chạy chi tiết ở Chương 5."),
    ("rp01_pipeline", "3.1", "Tiền xử lý bộ dữ liệu khoảng 700 giờ",
     "Sáu bước từ lúc quay chuyển động người tới lúc sẵn sàng huấn luyện SONIC",
     None,
     "Hình mô tả sáu bước đưa bộ dữ liệu khoảng 700 giờ, dùng để huấn luyện mô hình trong bài báo SONIC và nhiều khả năng là "
     "bộ Bones RP01, từ lúc quay tới lúc sẵn sàng huấn luyện. Hai bước đầu do Bones Studio thực hiện: hơn 170 diễn viên gắn điểm "
     "đánh dấu được quay bằng hệ thống Vicon ở 120 khung hình mỗi giây, với hơn 20 phong cách diễn; chuyển động của mọi diễn viên "
     "sau đó được đưa về một khung xương người chung, lưu ở dạng BVH và FBX, tổng khoảng 700 giờ. Ba bước tiếp theo do NVIDIA thực "
     "hiện theo bài báo: chuyển tư thế người sang robot G1 bằng GMR và PyRoki, được 29 góc khớp ở mỗi khung; loại các động tác robot "
     "không làm được trên sàn phẳng như leo cầu thang hay ngồi, còn 611 giờ; thêm bản lật gương trái–phải và chia thành một tập "
     "huấn luyện 317.189 clip cùng hai tập kiểm tra. Bước cuối, vẽ nét đứt, không có trong bài báo mà được suy ra từ mã nguồn công "
     "khai: mỗi clip được lưu thành một tệp motion library ở 30 khung mỗi giây, rồi nội suy lên 50 khung mỗi giây khi huấn luyện. "
     "Dữ liệu sau bước này được dùng để huấn luyện SONIC và Kinematic Planner."),
    ("seed_prep", "3.2", "Chuẩn bị Bones-SEED để fine-tune SONIC",
     "Phần robot do nhóm thực hiện tự chuyển định dạng và lọc; phần người NVIDIA đã xử lý sẵn",
     None,
     "Khác với bộ 700 giờ, phần lớn việc tiền xử lý Bones-SEED đã được làm trước khi phát hành, nên nhóm thực hiện chỉ cần bốn "
     "bước. Hàng trên là phần robot, do nhóm tự xử lý. Bước 1 chuyển 142.220 tệp CSV chuyển động robot G1 sang định dạng motion "
     "library mà SONIC đọc được, đồng thời hạ tần số từ 120 xuống 30 khung mỗi giây; kết quả là 142.220 tệp PKL. Ở bước này cần "
     "khai báo đúng tần số nguồn 120 khung mỗi giây; nếu khai báo sai, chuyển động bị chậm đi 2,4 lần mà công cụ chuyển đổi không "
     "đưa ra cảnh báo. Bước 2 loại các clip robot không làm được bằng cách so tên tệp với 40 từ khoá, như leo cầu thang, ngồi hay "
     "ghế; còn lại 129.785 clip, tương đương 264,5 giờ. Hàng dưới là phần người: NVIDIA đã chuyển sẵn các clip sang dạng SMPL, nên "
     "ở bước 3 chỉ cần tải khoảng 30 GB từ Hugging Face và giải nén, được khoảng 131 nghìn tệp. Ở bước 4, mỗi clip robot được ghép "
     "với dữ liệu người có cùng tên clip để fine-tune SONIC từ checkpoint phát hành. Clip robot không có dữ liệu người tương ứng vẫn "
     "được dùng, nhưng phần dữ liệu người được điền bằng 0."),
    ("ft_flow", "3.3", "Quy trình fine-tune SONIC trên Bones-SEED",
     "Chuẩn bị và kiểm tra dữ liệu, fine-tune, đánh giá, xuất ONNX rồi chạy cùng GR00T",
     None,
     "Theo tài liệu huấn luyện của GR00T-WholeBodyControl, quy trình fine-tune gồm sáu bước. Bước 1 chuẩn bị dữ liệu như hình "
     "trước. Bước 2 phát lại một vài clip trong mô phỏng để kiểm tra robot chuyển động đúng tốc độ và bàn chân không lún xuống dưới mặt sàn, trước khi "
     "dùng tài nguyên để huấn luyện. Bước 3 fine-tune từ checkpoint phát hành với learning rate nhỏ, 5 × 10⁻⁶, và lưu checkpoint "
     "sau mỗi 2000 bước; mỗi bước huấn luyện diễn ra như Hình 4.1. Bước 4 đánh giá checkpoint theo ba chỉ số: tỉ lệ thành công lớn "
     "hơn 0,97, MPJPE cục bộ nhỏ hơn 30 mm và MPJPE toàn cục nhỏ hơn 200 mm; nếu chưa đạt, cấu hình được điều chỉnh và huấn luyện "
     "tiếp. Khi đạt, bước 5 xuất encoder và decoder sang ONNX để chương trình C++ trên robot sử dụng, và bước 6 chạy checkpoint mới "
     "trong mô phỏng hoặc trên robot thật, ghép với GR00T. Điểm cần lưu ý khi dùng với GR00T: mỗi checkpoint có không gian token "
     "riêng, nên sau khi fine-tune, token tư thế ban đầu và nhãn motion token trong dữ liệu fine-tune GR00T phải được tạo lại bằng "
     "checkpoint mới. Về tài nguyên, tài liệu khuyến nghị từ 64 GPU trở lên; máy 8 GPU vẫn chạy được nhưng chậm hơn đáng kể."),
    ("sonic_train", "4.1", "Một bước huấn luyện SONIC",
     "Ba giai đoạn: tạo token, điều khiển robot mô phỏng, chấm điểm và học",
     ["<b>①</b> Encoder và FSQ nén clip mẫu thành token.",
      "<b>②</b> Decoder g1_dyn biến token và trạng thái robot thành góc đích; robot trong Isaac Lab thực hiện theo.",
      "<b>③</b> So chuyển động với clip mẫu để chấm điểm; PPO dùng chênh lệch với ước lượng của Critic để cập nhật Actor.",
      "Actor của PPO gồm encoder, FSQ và hai decoder, huấn luyện đồng thời.",
      "Decoder g1_kin chỉ dựng lại chuyển động để tính loss, không điều khiển robot."],
     "Hai khối nét đứt, decoder g1_kin và Critic, chỉ dùng khi huấn luyện và không có trong checkpoint phát hành. "
     "Decoder g1_kin chỉ dựng lại chuyển động từ token để tính loss dựng lại; đầu ra của nó không điều khiển robot, "
     "không đưa vào Isaac Lab, và trọng số của nó chỉ được cập nhật bằng loss đó. PPO cập nhật encoder và decoder g1_dyn."),
    ("enc_window", "4.2", "Các khung encoder đọc từ clip",
     "Encoder G1 đọc 10 khung phủ 0,9 s; encoder SMPL đọc 10 khung liền nhau phủ 0,18 s",
     None,
     "Phần chân của encoder teleop dùng cùng các khung với encoder G1, ba điểm VR chỉ lấy khung tại t. "
     "Sang bước sau, t tăng 0,02 s và cả cửa sổ trượt theo."),
    ("fsq", "4.3", "Cách tạo motion token",
     "2 token × 32 chiều, mỗi chiều 32 nấc; decoder và GR00T chỉ thấy dãy 64 số nối liền",
     None,
     "Số 32 xuất hiện hai lần với hai nghĩa: mỗi token có 32 ô, và mỗi ô chỉ nhận một trong 32 giá trị. "
     "Các giá trị trong hình chỉ để minh hoạ."),
    ("prop_window", "4.4", "Proprioception mà decoder nhận",
     "10 khung gần nhất × 93 số = 930 số; cộng token 64 chiều thành 994 số đầu vào",
     None,
     "Cửa sổ luôn là 0,2 giây vừa qua. Khác với encoder nhìn về tương lai, decoder nhìn về quá khứ."),
    ("combine", "5.1", "Luồng chạy khi ghép GR00T với SONIC",
     "Vòng ngoài ≤2,5 Hz: GR00T quyết định làm gì · Vòng trong 50 Hz: SONIC giữ thăng bằng và bám token",
     None,
     "Góc ngón tay không đi qua SONIC mà đi thẳng tới bàn tay Dex3."),
    ("runtime_detail", "5.2", "Luồng chạy chi tiết",
     "Các luồng, tần số và cổng ZMQ giữa máy GPU, client Python và robot",
     ["<b>Máy GPU</b>: GR00T server, ≤2,5 Hz, trả về chunk 40 bước.",
      "<b>Client Python</b>: luồng suy luận giữ chunk mới nhất; vòng chính bù trễ, gửi một bước mỗi 20 ms.",
      "<b>Robot G1</b>: decoder SONIC chạy 50 Hz, ghi lệnh 500 Hz cho 29 động cơ thân; Dex3 nhận góc tay.",
      "Nét đứt: trạng thái robot gửi ngược về client."],
     "Theo mã nguồn client và chương trình triển khai của NVIDIA. Camera server chạy trên robot, vẽ tách riêng cho dễ theo dõi."),
    ("timeline", "5.3", "Dòng thời gian của client, độ trễ Δ = 0,1 s",
     "Client gọi lần mới 0,4 s sau khi nhận chunk, nên hai lần gọi cách nhau 0,5 s",
     None,
     "Khi nhận chunk mới, client thay ngay chunk cũ và bắt đầu từ bước 5; các bước 30 đến 39 của chunk A không bao giờ được gửi."),
    ("label_flow", "5.4", "Nguồn của motion token",
     "Quy trình do NVIDIA đề xuất: lưu token SONIC dùng khi người điều khiển robot làm đáp án, rồi dạy GR00T đoán ra token đó",
     None,
     "GR00T được fine-tune trên các token ghi làm nhãn. Decoder SONIC giữ nguyên ở cả hai trường hợp. "
     "Quy trình do NVIDIA đề xuất (bài báo SONIC Mục 2.4–2.5, Phụ lục S1.4; tài liệu VLA Workflow của GR00T-WholeBodyControl); hình do nhóm vẽ lại. "
     "Việc tự tính nhãn cho dữ liệu mô phỏng là hướng mở rộng do nhóm đề xuất."),
]

# Slide video Kinematic Planner (MotionBricks), sinh bằng tools/render_motionbricks_planner.py.
# (tên, số hình, tiêu đề, phụ đề, các ý, ghi chú, ảnh tĩnh, video) — ảnh và video trong assets/.
MB_VIDEO = ("mb_video", "1.4", "MotionBricks: Kinematic Planner của SONIC",
            "Từ cùng một tư thế đứng, bốn lệnh cho bốn đoạn chuyển động khác nhau",
            ["<b>Đầu vào</b>: 4 khung tư thế gần nhất và lệnh đơn giản: kiểu chuyển động, hướng, tốc độ, độ cao.",
             "<b>Đầu ra</b>: chuỗi tư thế 24–64 khung ở 30 Hz, tức 0,8–2,1 s; cứ khoảng 1 s planner sinh tiếp đoạn mới.",
             "Đoạn chuyển tiếp như từ đứng sang chạy, từ đứng cúi xuống bò do mô hình tự sinh, không quay sẵn.",
             "Chỉ là chuyển động động học: không có vật lý. SONIC nhận chuyển động này và tự giữ thăng bằng khi làm theo."],
            "Video do nhóm thực hiện tạo bằng cách chạy tệp planner NVIDIA phát hành trên CPU, mỗi ô ghép ba lần gọi planner, "
            "rồi đặt tư thế vào mô hình G1 trong MuJoCo để hiển thị, không mô phỏng vật lý. Trong 3 giây đầu robot đi được 1,8 m, "
            "chạy được 4,6 m. Khi GR00T điều khiển robot, planner không chạy; planner chỉ dùng khi điều khiển bằng tay cầm, "
            "bàn phím, hoặc để sinh chân khi teleop VR ba điểm.",
            "motionbricks/planner_demo_still.png", "motionbricks/planner_demo.mp4")

# Phần giải thích cạnh hình cho các hình TikZ chưa có ý chính trong FIGS: tên hình -> (bố cục, các ý).
# Mọi hình TikZ dùng bố cục "tall": tiêu đề và giải thích ở cột trái, hình ở bên phải.
EXPLAIN = {
    "gr00t_arch": ("tall", [
        "<b>System 2</b>: VLM Cosmos-Reason2-2B đổi ảnh và câu lệnh thành 116 token (96 ảnh, 13 chữ, 7 định dạng); đóng băng khi fine-tune.",
        "<b>System 1</b>: trạng thái thành 1 token, 40 bước hành động có nhiễu thành 40 token; DiT 16 khối xen kẽ tự chú ý và chú ý chéo tới token VLM.",
        "Khử nhiễu 4 lần, ra chunk 40 bước; mỗi bước 78 số: 64 token SONIC + 14 góc ngón tay.",
        "Khối xanh lá có trọng số riêng cho từng embodiment và được fine-tune."]),
    "rp01_pipeline": ("tall", [
        "<b>Bước 1–2, Bones Studio</b>: hơn 170 diễn viên, Vicon 120 khung/giây, đưa về một khung xương chung; khoảng 700 giờ.",
        "<b>Bước 3–5, NVIDIA</b>: retarget sang G1 bằng GMR và PyRoki; loại động tác không làm được, còn 611 giờ; lật gương, chia tập, 317.189 clip huấn luyện.",
        "<b>Bước 6</b> (nét đứt, suy ra từ mã nguồn): lưu motion library 30 khung/giây, nội suy lên 50 khi huấn luyện."]),
    "seed_prep": ("tall", [
        "<b>Phần robot, nhóm tự làm</b>: chuyển 142.220 tệp CSV sang motion library, hạ 120 xuống 30 khung/giây.",
        "<b>Lọc 40 từ khoá</b> (leo cầu thang, ngồi, ghế…): còn 129.785 clip, 264,5 giờ.",
        "<b>Phần người</b>: tải bản SMPL NVIDIA phát hành, khoảng 30 GB, khoảng 131 nghìn tệp.",
        "<b>Bước 4</b>: ghép robot với người theo tên clip, fine-tune từ checkpoint phát hành."]),
    "ft_flow": ("tall", [
        "Chuẩn bị dữ liệu, rồi phát lại vài clip để kiểm tra: đúng tốc độ, bàn chân không lún xuống dưới mặt sàn.",
        "Fine-tune từ checkpoint phát hành, learning rate 5 × 10⁻⁶, lưu checkpoint mỗi 2000 bước.",
        "Đánh giá: thành công > 0,97, MPJPE cục bộ < 30 mm; chưa đạt thì chỉnh cấu hình, học tiếp.",
        "Đạt thì xuất ONNX và chạy cùng GR00T; nhãn token của GR00T phải tạo lại bằng checkpoint mới."]),
    "enc_window": ("tall", [
        "Clip 50 khung/giây, mỗi vạch là 1 khung. Đường <b>t</b> là khung hiện tại, bên phải là các khung tương lai.",
        "<b>Encoder G1</b>: 10 khung cách 0,1 s, nhìn trước 0,9 s. <b>Encoder SMPL</b>: 10 khung liền nhau, nhìn trước 0,18 s.",
        "Khung tương lai là các tư thế robot cần đạt tiếp theo; lúc huấn luyện cả clip có sẵn nên đọc được. Mỗi bước t tăng 0,02 s, cửa sổ trượt theo."]),
    "fsq": ("tall", [
        "Encoder đọc khoảng 1 giây chuyển động, xuất 64 số thực.",
        "Xếp thành 2 token × 32 ô; FSQ làm tròn mỗi ô về 1 trong 32 nấc, cách nhau 1/16.",
        "Nối lại thành dãy 64 số: đây là motion token mà decoder nhận và GR00T học dự đoán.",
        "Số 32 có hai nghĩa: 32 ô mỗi token, và 32 giá trị mỗi ô."]),
    "prop_window": ("tall", [
        "Decoder đọc <b>10 khung gần nhất</b>, cách nhau 0,02 s, tức 0,2 giây vừa qua.",
        "Mỗi khung 93 số: vận tốc góc thân 3, góc khớp 29, vận tốc khớp 29, lệnh trước 29, trọng lực 3.",
        "10 × 93 = <b>930</b> số; cộng token 64 thành <b>994</b> số đầu vào decoder.",
        "Decoder không có bộ nhớ, nên cần lịch sử để biết robot đang ngã về đâu."]),
    "combine": ("tall", [
        "<b>Vòng ngoài, tối đa 2,5 lần/giây</b>: GR00T nhận ảnh, câu lệnh, trạng thái 46 chiều, xuất chunk 40 bước.",
        "<b>Client</b> bù trễ, gửi 1 bước mỗi 20 ms: token 64 tới SONIC, góc tay 14 tới bàn tay Dex3.",
        "<b>Vòng trong 50 Hz</b>: decoder SONIC ghép token với proprioception 930 số, ra 29 góc đích; PD trong driver nhận lệnh 500 Hz."]),
    "timeline": ("tall", [
        "Ví dụ độ trễ Δ = 0,1 s: client gọi GR00T, 0,1 s sau mới nhận chunk A.",
        "Chunk đã cũ 0,1 s, tức 5 bước, nên client gửi từ A[5]; gọi lại 0,4 s sau khi nhận, nên hai lần gọi cách 0,5 s.",
        "Chunk B về thì chuyển ngay sang B[5]; bước 30–39 của A không bao giờ được gửi. Với cấu hình mặc định, Δ cần dưới khoảng 0,2 s."]),
    "label_flow": ("tall", [
        "<b>Thu dữ liệu</b>: người đeo kính VR điều khiển; encoder teleop sinh token, token được ghi làm nhãn.",
        "<b>Fine-tune</b>: GR00T học đoán đúng token đó từ ảnh, câu lệnh và trạng thái.",
        "<b>Lúc chạy</b>: GR00T thay người và encoder; decoder g1_dyn giữ nguyên, nên nhãn phải dùng đúng checkpoint SONIC.",
        "<b>Do NVIDIA đề xuất</b>: bài báo SONIC fine-tune GR00T N1.5 theo cách này, đạt 75% trung bình trên 5 nhiệm vụ; tài liệu VLA của NVIDIA áp dụng cho GR00T N1.7."]),
}


def with_explain(f):
    """Gắn bố cục và các ý giải thích (nếu có trong EXPLAIN) vào một mục của FIGS."""
    if f[0] in EXPLAIN and not f[4]:
        layout, points = EXPLAIN[f[0]]
        return (*f[:4], points, f[5], layout)
    return f

# Biểu đồ thống kê Bones-SEED (tệp png trong figures/bones_seed/). Cùng cấu trúc với FIGS;
# bố cục "side": biểu đồ bên trái, nhận xét bên phải; có "tall" thì chữ trái, hình phải như FIGS.
STATS = [
    ("package", "2.1", "Số clip và số giờ theo package",
     "Locomotion chiếm khoảng 52%, cả theo số clip lẫn theo số giờ",
     ["<b>Locomotion</b>: 74.488 clip, khoảng 52% cả theo số clip lẫn số giờ.",
      "<b>Interactions</b> (cầm nắm, gắp đặt, mang vác) gần bài toán thao tác nhất: 14.643 clip.",
      "Độ dài clip trung bình mỗi package chỉ từ 5,6 s (Communication) đến 10,7 s (Everyday), nên tỉ lệ số giờ gần bằng tỉ lệ số clip."],
     "Tính từ tệp metadata phiên bản v004, gồm cả clip lật gương.", "side"),
    ("category", "A.1", "Số clip theo 20 category",
     "Lớn nhất là đi lại phong cách bình thường, 33.430 clip; nhỏ nhất là võ thuật, 20 clip",
     ["8 package được chia thành 20 category; RP01, bộ dữ liệu của bài báo, có 33 nhóm lớn và 8.447 nhóm con.",
      "15 trong 16 category của RP01 mà Bones Studio công bố có trong Bones-SEED; chỉ Combat là không có.",
      "Khi huấn luyện SONIC, robot chỉ học động tác cơ thể; trong mô phỏng không có đồ vật."],
     "Số clip theo 20 category, gồm cả clip lật gương.", "side"),
    ("duration", "2.2", "Phân bố độ dài clip",
     "Phần lớn clip ngắn: trung bình 7,3 s, trung vị 5,7 s",
     ["83% clip dưới 10 s; ngắn nhất 0,6 s, dài nhất 180 s.",
      "Chỉ 1.448 clip dài hơn 30 s, được gộp vào cột cuối.",
      "Tổng 288,3 giờ: 144,2 giờ quay thật, nửa còn lại là bản lật gương."],
     "Trục hoành cắt ở 30 s; các clip dài hơn được gộp vào cột cuối.", "side"),
    ("movement", "2.3", "Kiểu chuyển động và tư thế cơ thể",
     "Đi bộ, chạy bộ và cử chỉ phổ biến nhất; tư thế đứng chiếm 86,2% số clip",
     ["Metadata có 150 kiểu chuyển động và 57 tư thế khác nhau.",
      "Ngồi sàn, ngồi xổm, bò và quỳ cộng lại chỉ vài phần trăm.",
      "Nhãn tư thế có lỗi chính tả: “standing, croaching” và “standing, crouching” cần gộp khi thống kê."],
     "Phần trăm tính trên toàn bộ 142.220 clip.", "side"),
    ("flags", "A.2", "Các cờ nội dung trong metadata",
     "19,4% clip có đạo cụ hoặc đồ vật; 8% clip có phong cách khác bình thường",
     ["Đồ vật thường gặp: hộp 50 cm, thùng, cửa, kệ, ghế.",
      "Phong cách khác bình thường chủ yếu là giả bị thương ở chân hoặc thân.",
      "Cờ di chuyển ngang bật ở 72%, cờ dọc ở 64% số clip; thẻ dữ liệu không nêu tiêu chí, và cờ dọc bật cả ở clip đứng yên."],
     "Tỉ lệ clip theo các cờ nội dung trong metadata.", "side"),
    ("actors", "A.3", "Thống kê 522 diễn viên",
     "Gần cân bằng giới tính: 269 nam, 253 nữ",
     ["Từ 17 đến 71 tuổi; 85% không quá 35 tuổi, chỉ 32 người từ 50 tuổi trở lên.",
      "Chiều cao 145–199 cm, cân nặng 38–145 kg.",
      "Số clip mỗi người từ 2 đến 899; 10% diễn viên đóng góp nhiều nhất chiếm 27% số clip.",
      "Với SONIC, chênh lệch này ít ảnh hưởng vì dữ liệu G1 đã được retarget về cùng một robot."],
     "Mỗi diễn viên tính một lần: (a) tuổi, (b) chiều cao, (c) cân nặng, (d) giới tính, (e) nghề, (f) số clip của mỗi diễn viên.",
     "tall"),
    ("timeline", "A.4", "Số clip theo tháng quay",
     "Quay từ 05/2021 đến 09/2024, trong 124 ngày quay",
     ["98,6% số clip được quay trong khoảng 07/2022 đến 11/2023.",
      "Buổi quay cuối 18/09/2024; mã nguồn NVIDIA nhắc tới một thư mục dữ liệu Bones nội bộ ngày 24/09/2024, sáu ngày sau.",
      "Đây là một dấu hiệu Bones-SEED và dữ liệu NVIDIA dùng lấy từ cùng các buổi quay, chưa đủ để khẳng định là tập con."],
     "Số clip theo tháng quay.", "side"),
]

# Ô số liệu ở slide tổng quan Bones-SEED: (số, đơn vị, dòng giải thích)
OVERVIEW = [
    ("142.220", "clip", "71.132 clip quay thật và 71.088 clip lật gương"),
    ("288,3", "giờ", "144,2 giờ quay thật; tính cả bản lật gương"),
    ("522", "diễn viên", "253 nữ, 269 nam; 17 đến 71 tuổi"),
    ("8", "nhóm động tác lớn (package)", "Chia tiếp thành 20 nhóm nhỏ (category); Locomotion chiếm khoảng 52% số clip"),
    ("120", "khung mỗi giây", "SONIC huấn luyện ở 50 Hz"),
    ("3", "định dạng chuyển động", "SOMA Proportional BVH, SOMA Uniform BVH, Unitree G1 CSV"),
]

CHROME = f"""<div style="position:absolute; left:0; top:0; width:1920px; height:28px; background:#1A4A35"></div>
<div style="position:absolute; left:0; top:28px; width:1920px; height:10px; background:#2DB36A"></div>
<div style="position:absolute; left:0; top:38px; width:56px; height:1042px; background:{{strip}}"></div>
<div style="position:absolute; left:56px; top:38px; width:7px; height:1042px; background:#2DB36A"></div>
{{logo_box}}<img src="{LOGO}" alt="VinDynamics" style="position:absolute; left:120px; top:66px; width:200px; height:110px; object-fit:contain">
<p style="position:absolute; right:128px; top:100px; width:900px; text-align:right; font-size:24px; font-weight:700; color:#1C8A4B; letter-spacing:1px; text-transform:uppercase">{{label}}</p>
<div style="position:absolute; left:120px; top:984px; width:320px; height:12px; background:#2DB36A"></div>
<div style="position:absolute; left:456px; top:984px; width:110px; height:12px; background:#1A4A35"></div>
<p style="position:absolute; left:120px; bottom:24px; width:600px; font-size:24px; color:#6E7C75">VinDynamics  |  Confidential</p>
"""
LOGO_BOX = '<div style="position:absolute; left:112px; top:62px; width:216px; height:118px; background:#FEFFFE; border-radius:14px"></div>\n'
PAGE = '<p style="position:absolute; right:128px; bottom:24px; width:200px; text-align:right; font-size:32px; font-weight:700; color:#173F2E">{:02d}</p>\n'


def svg_ratio(name):
    head = (TIKZ / f"{name}.svg").read_text(encoding="utf-8")[:600]
    w = float(re.search(r'width="([\d.]+)', head).group(1))
    h = float(re.search(r'height="([\d.]+)', head).group(1))
    return w / h


def png_ratio(path):
    w, h = struct.unpack(">II", path.read_bytes()[16:24])  # khối IHDR của PNG
    return w / h


def fit(ratio, box_w, box_h):
    """Kích thước ảnh lớn nhất vừa khung, giữ tỉ lệ."""
    w, h = box_w, box_w / ratio
    if h > box_h:
        w, h = box_h * ratio, box_h
    return int(w), int(h)


def bullets(points, size=23):
    return "\n".join(
        f'<div style="display:flex; flex-direction:row; gap:14px; align-items:start">'
        f'<div style="width:12px; height:12px; margin-top:{int(size * 0.48)}px; border-radius:3px; background:#2DB36A; flex:none"></div>'
        f'<p style="font-size:{size}px; line-height:1.4; color:#4B5B53">{p}</p></div>' for p in points)


def figure_slide(page, name, num, title, sub, points, note, layout=None, src=None, ratio=None, video=None):
    key = num.split(".")[0]
    chap = CHAPTERS[int(key) if key.isdigit() else key]
    head = CHROME.format(strip="#E3F0E8", logo_box=LOGO_BOX, label=f"Hình {num} · {chap}") + PAGE.format(page)
    src = src or f"/_asset/tikz/{name}.svg"
    ratio = ratio or svg_ratio(name)
    layout = layout or ("tall" if points else "wide")
    # video: ảnh tĩnh ở src, video ở data-video (bản trình chiếu phát video, bản PDF chỉ in ảnh tĩnh)
    vid = f' data-video="{video}"' if video else ""
    img = '<img src="' + src + '"' + vid + ' alt="Hình {num}: {t}" style="width:{w}px; height:{h}px; object-fit:contain">'
    if layout == "side":
        # Biểu đồ bên trái, nhận xét bên phải
        w, h = fit(ratio, 1182, 586)
        body = f"""<div style="display:flex; flex-direction:column; gap:10px">
<h2 style="font-size:54px; font-weight:800; line-height:1.15; color:#173F2E; text-transform:uppercase">{title}</h2>
<p style="font-size:28px; color:#5D6B64">{sub}</p>
</div>
<div style="display:flex; flex-direction:row; gap:40px; align-items:start">
<div style="width:1232px; height:636px; background:#FEFFFE; border:1px solid #E2E9E5; border-radius:24px; padding:24px; display:flex; align-items:center; justify-content:center; flex:none">
{img.format(num=num, t=title, w=w, h=h)}
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:20px; padding-top:12px">
<p style="font-size:24px; font-weight:700; color:#1C8A4B; letter-spacing:1px; text-transform:uppercase">{"Nhận xét" if src and src.startswith("/_fig/") else "Giải thích"}</p>
<div style="width:120px; height:6px; border-radius:3px; background:#2DB36A"></div>
{bullets(points, 23)}
</div>
</div>"""
        sec = (f'<section id="fig{num.replace(".", "_")}" data-transition="fade" style="background:#F3F8F5; color:#4B5B53; {FONT}; '
               f'padding:200px 128px 116px 120px; display:flex; flex-direction:column; gap:24px">')
    elif layout == "tall":
        # Hình gần vuông: chữ bên trái, hình bên phải chiếm hết chiều cao
        w, h = fit(ratio, 1050, 704)
        items = bullets(points, 23)
        body = f"""<div style="flex:1; display:flex; flex-direction:column; gap:26px">
<div style="display:flex; flex-direction:column; gap:12px">
<h2 style="font-size:46px; font-weight:800; line-height:1.15; color:#173F2E; text-transform:uppercase">{title}</h2>
<p style="font-size:26px; line-height:1.4; color:#5D6B64">{sub}</p>
</div>
<div style="width:120px; height:6px; border-radius:3px; background:#2DB36A"></div>
<div style="display:flex; flex-direction:column; gap:14px">
{items}
</div>
</div>
<div style="width:1100px; height:754px; background:#FEFFFE; border:1px solid #E2E9E5; border-radius:24px; padding:24px; display:flex; align-items:center; justify-content:center; flex:none">
{img.format(num=num, t=title, w=w, h=h)}
</div>"""
        sec = (f'<section id="fig{num.replace(".", "_")}" data-transition="fade" style="background:#F3F8F5; color:#4B5B53; {FONT}; '
               f'padding:200px 128px 116px 120px; display:flex; flex-direction:row; gap:48px; align-items:start">')
    else:
        w, h = fit(ratio, 1622, 586)
        body = f"""<div style="display:flex; flex-direction:column; gap:10px">
<h2 style="font-size:54px; font-weight:800; line-height:1.15; color:#173F2E; text-transform:uppercase">{title}</h2>
<p style="font-size:28px; color:#5D6B64">{sub}</p>
</div>
<div style="width:1672px; height:636px; background:#FEFFFE; border:1px solid #E2E9E5; border-radius:24px; padding:24px; display:flex; align-items:center; justify-content:center; flex:none">
{img.format(num=num, t=title, w=w, h=h)}
</div>"""
        sec = (f'<section id="fig{num.replace(".", "_")}" data-transition="fade" style="background:#F3F8F5; color:#4B5B53; {FONT}; '
               f'padding:200px 128px 116px 120px; display:flex; flex-direction:column; gap:24px">')
    return f"{sec}\n{head}{body}\n<aside>{note}</aside>\n</section>\n"


def overview_slide(page):
    head = CHROME.format(strip="#E3F0E8", logo_box=LOGO_BOX, label="Bộ dữ liệu Bones-SEED") + PAGE.format(page)
    cards = "\n".join(
        f'<div style="width:520px; height:228px; background:#FEFFFE; border:1px solid #E2E9E5; border-radius:24px; '
        f'padding:26px 36px; display:flex; flex-direction:column; gap:8px; flex:none">'
        f'<div style="width:64px; height:6px; border-radius:3px; background:#2DB36A; flex:none"></div>'
        f'<p style="font-size:66px; font-weight:800; line-height:1.05; color:#173F2E">{n}</p>'
        f'<p style="font-size:26px; font-weight:700; color:#1C8A4B">{u}</p>'
        f'<p style="font-size:22px; line-height:1.35; color:#5D6B64">{d}</p></div>' for n, u, d in OVERVIEW)
    return f"""<section id="bones_overview" data-transition="fade" style="background:#F3F8F5; color:#4B5B53; {FONT}; padding:200px 128px 116px 120px; display:flex; flex-direction:column; gap:28px">
{head}<div style="display:flex; flex-direction:column; gap:10px">
<h2 style="font-size:54px; font-weight:800; line-height:1.15; color:#173F2E; text-transform:uppercase">Tổng quan bộ dữ liệu Bones-SEED</h2>
<p style="font-size:28px; color:#5D6B64">Phần công khai của thư viện chuyển động Bones Studio, đã được NVIDIA chuyển sang robot G1</p>
</div>
<div style="display:flex; flex-direction:row; flex-wrap:wrap; gap:24px 56px">
{cards}
</div>
<p style="font-size:24px; line-height:1.4; font-style:italic; color:#6E7C75; border-left:4px solid #2DB36A; padding:2px 0px 2px 18px">Số liệu từ thẻ dữ liệu trên Hugging Face và tệp metadata v004 (51 cột, khoảng 4,5 MB). Bộ dữ liệu không có ảnh, video, âm thanh hay lực tiếp xúc.</p>
<aside>Bones-SEED gồm 142.220 clip, khoảng 288 giờ tính cả bản lật gương. Mỗi clip có chuyển động ở ba định dạng, kèm mô tả ngôn ngữ, nhãn phân đoạn và metadata. SONIC chỉ dùng chuyển động robot G1 làm dữ liệu chính.</aside>
</section>
"""


def cover_slide():
    head = CHROME.format(strip="#EEF6F1", logo_box="", label="Báo cáo kỹ thuật · 10/2026")
    groups = [("01", "Kiến trúc GR00T và SONIC", "Hình 1.1 – 1.3"),
              ("02", "Thống kê Bones-SEED", "Hình 2.1 – 2.3, A.1 – A.4"),
              ("03", "Tiền xử lý dữ liệu", "Hình 3.1 – 3.3"),
              ("04", "Huấn luyện SONIC", "Hình 4.1 – 4.4"),
              ("05", "Ghép GR00T với SONIC", "Hình 5.1 – 5.4")]
    rows = "\n".join(
        f'<div style="display:flex; flex-direction:row; gap:28px; align-items:center">'
        f'<p style="width:96px; font-size:52px; font-weight:800; color:#2DB36A">{n}</p>'
        f'<div style="display:flex; flex-direction:column; gap:4px">'
        f'<p style="font-size:34px; font-weight:700; color:#173F2E">{t}</p>'
        f'<p style="font-size:24px; color:#6E7C75">{f}</p></div></div>' for n, t, f in groups)
    return f"""<section id="cover" data-transition="fade" style="background:#FEFFFE; color:#4B5B53; {FONT}; padding:200px 128px 150px 120px; display:flex; flex-direction:row; align-items:center; gap:64px">
{head}<div style="flex:1; display:flex; flex-direction:column; gap:28px">
<p style="font-size:26px; font-weight:700; color:#1C8A4B; letter-spacing:1px; text-transform:uppercase">Hình minh hoạ của báo cáo</p>
<h1 style="font-size:84px; font-weight:800; line-height:1.1; color:#173F2E">Bones-SEED, GEAR-SONIC<br><span style="color:#2DB36A">×</span> GR00T N1.7</h1>
<p style="font-size:32px; line-height:1.4; color:#5D6B64">Dữ liệu Bones-SEED, cách GEAR-SONIC xử lý và ghép nối với GR00T N1.7</p>
<hr style="width:640px; border-top:2px solid #E2E9E5">
<div style="display:flex; flex-direction:column; gap:8px">
<p style="font-size:30px; font-weight:700; color:#173F2E">Nguyễn Duy Hoàng · Phạm Thanh Hưng</p>
<p style="font-size:26px; color:#6E7C75">Mentor: Trần Ngọc Sơn, Nguyễn Văn Thái</p>
</div>
</div>
<div style="width:8px; height:600px; background:#2DB36A"></div>
<div style="width:700px; height:600px; background:#F4F7F5; border:1px solid #E2E9E5; border-radius:24px; display:flex; flex-direction:column; justify-content:center; gap:26px; padding:40px 56px">
<p style="font-size:24px; font-weight:700; color:#1C8A4B; letter-spacing:1px; text-transform:uppercase">Nội dung</p>
{rows}
</div>
<aside>Bộ slide gồm 14 hình TikZ và các biểu đồ thống kê Bones-SEED của báo cáo kỹ thuật, xếp theo thứ tự chương.</aside>
</section>
"""


def thanks_slide(page):
    src = (HERE / "deck" / "slides" / "thanks.html").read_text(encoding="utf-8")
    return re.sub(r'(border-radius:16px">)\d+(</p>)', rf"\g<1>{page:02d}\g<2>", src, count=1)


def main():
    (OUT / "slides").mkdir(parents=True, exist_ok=True)
    order = ["cover"]
    # Các hình Chương 1, rồi phần thống kê Bones-SEED (Chương 2 và Phụ lục A), rồi các hình TikZ còn lại
    figs = [with_explain(f) for f in FIGS]
    ch1 = [f for f in figs if f[1].startswith("1.")]
    slides = [(f"fig{f[1].replace('.', '_')}", lambda p, f=f: figure_slide(p, *f)) for f in ch1]
    still = HERE / "assets" / MB_VIDEO[6]
    slides.append(("mb_video", lambda p: figure_slide(p, *MB_VIDEO[:6], layout="tall", src=f"/_asset/{MB_VIDEO[6]}",
                                                     ratio=png_ratio(still), video=f"/_asset/{MB_VIDEO[7]}")))
    slides.append(("bones_overview", overview_slide))
    for name, num, title, sub, points, note, layout in STATS:
        path = FIG / "bones_seed" / f"{name}.png"
        slides.append((f"fig{num.replace('.', '_')}",
                       lambda p, a=(name, num, title, sub, points, note, layout, f"/_fig/bones_seed/{name}.png", png_ratio(path)):
                       figure_slide(p, *a)))
    for f in figs[len(ch1):]:
        slides.append((f"fig{f[1].replace('.', '_')}", lambda p, f=f: figure_slide(p, *f)))
    for old in (OUT / "slides").glob("*.html"):
        old.unlink()  # chỉ xoá slide do script này sinh ra trong deck_tikz/slides
    for page, (sid, make) in enumerate(slides, start=2):
        (OUT / "slides" / f"{sid}.html").write_text(make(page), encoding="utf-8")
        order.append(sid)
    (OUT / "slides" / "cover.html").write_text(cover_slide(), encoding="utf-8")
    (OUT / "slides" / "thanks.html").write_text(thanks_slide(len(slides) + 2), encoding="utf-8")
    order.append("thanks")
    main_deck = json.loads((HERE / "deck" / "deck.json").read_text(encoding="utf-8"))
    deck = {"v": main_deck.get("v", 4), "title": "Hình TikZ: Bones-SEED, GEAR-SONIC và GR00T N1.7",
            "order": order, "faces": main_deck["faces"]}
    (OUT / "deck.json").write_text(json.dumps(deck, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(order)} slide -> {OUT}")


if __name__ == "__main__":
    main()
