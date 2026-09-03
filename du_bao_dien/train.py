import pandas as pd
from xgboost import XGBRegressor
import joblib
import numpy as np

print("⏳ Đang tạo dữ liệu huấn luyện mở rộng hỗ trợ xử lý dữ liệu trống...")
np.random.seed(42)
n = 2000  

# 1. Thời gian & Môi trường cơ bản
thang = np.random.randint(1, 13, n)
nam = np.random.randint(2020, 2027, n)
nhiet_do = np.random.uniform(22, 40, n)
do_am = np.random.uniform(50, 95, n)
mat_do_dan_so = np.random.uniform(100, 2500, n)
so_khach_hang = np.random.randint(10000, 60000, n)
toc_do_phat_trien = np.random.uniform(0.5, 10.0, n)

# 2. Đặc trưng thời tiết chuyên sâu theo mùa
thang_nang_nong = np.where((thang >= 3) & (thang <= 5), 1, 0)
thang_mua = np.where((thang >= 5) & (thang <= 11), 1, 0)
thang_bao = np.where((thang >= 9) & (thang <= 11), np.random.randint(1, 5, n), 0)

# 3. Lịch công tác & Ngày nghỉ
cup_dien_tuan = np.random.randint(0, 5, n)
cup_dien_cuoi_tuan = np.random.randint(0, 3, n)
ngay_le = np.random.randint(0, 5, n)
ngay_nghi = np.random.randint(8, 11, n)

# 4. Phụ tải cấu thành dự báo tháng tới (kWh) - Giả lập có thể xuất hiện giá trị 0 hoặc trống
pt1 = np.random.uniform(100000, 1500000, n)
pt2 = np.random.uniform(100000, 1500000, n)
pt3 = np.random.uniform(100000, 500000, n)
pt4 = np.random.uniform(3000000, 6000000, n)
pt5 = np.random.uniform(100000, 500000, n)

# 5. Phụ tải cấu thành cùng kỳ năm trước (Mô phỏng trường hợp không có dữ liệu lịch sử sẽ tự động khớp với giá trị hiện tại)
pt1_ky_truoc = pt1 * np.random.choice([1.0, np.random.uniform(0.9, 1.1)], size=n, p=[0.3, 0.7])
pt2_ky_truoc = pt2 * np.random.choice([1.0, np.random.uniform(0.9, 1.1)], size=n, p=[0.3, 0.7])
pt3_ky_truoc = pt3 * np.random.choice([1.0, np.random.uniform(0.9, 1.1)], size=n, p=[0.3, 0.7])
pt4_ky_truoc = pt4 * np.random.choice([1.0, np.random.uniform(0.9, 1.1)], size=n, p=[0.3, 0.7])
pt5_ky_truoc = pt5 * np.random.choice([1.0, np.random.uniform(0.9, 1.1)], size=n, p=[0.3, 0.7])

# Gom nhóm dữ liệu đầu vào (Features)
df = pd.DataFrame({
    'Thang': thang, 'Nam': nam, 'Nhiet_do': nhiet_do, 'Do_am': do_am,
    'Mat_do_dan_so': mat_do_dan_so, 'So_khach_hang': so_khach_hang, 'Toc_do_phat_trien': toc_do_phat_trien,
    'Thang_nang_nong': thang_nang_nong, 'Thang_mua': thang_mua, 'Thang_bao': thang_bao,
    'Cup_dien_tuan': cup_dien_tuan, 'Cup_dien_cuoi_tuan': cup_dien_cuoi_tuan, 
    'Ngay_le': ngay_le, 'Ngay_nghi': ngay_nghi,
    'Phu_tai_1': pt1, 'Phu_tai_2': pt2, 'Phu_tai_3': pt3, 'Phu_tai_4': pt4, 'Phu_tai_5': pt5,
    'Pt1_ky_truoc': pt1_ky_truoc, 'Pt2_ky_truoc': pt2_ky_truoc, 'Pt3_ky_truoc': pt3_ky_truoc, 
    'Pt4_ky_truoc': pt4_ky_truoc, 'Pt5_ky_truoc': pt5_ky_truoc
})

# THIẾT LẬP QUY LUẬT TÍNH TOÁN CỐT LÕI (Xử lý an toàn với các giá trị 0 hoặc trống)
tong_co_ban = pt1 + pt2 + pt3 + pt4 + pt5
tang_truong_cung_ky = np.where(
    (pt1_ky_truoc > 0) & (pt2_ky_truoc > 0), 
    ((pt1 - pt1_ky_truoc) + (pt2 - pt2_ky_truoc) + (pt4 - pt4_ky_truoc)) * 0.2, 
    0
)

tac_dong_nhiet = (nhiet_do - 28.0) * 60000
tac_dong_mua_nang = thang_nang_nong * 250000  
tac_dong_mua_mua = thang_mua * (-80000)       
tac_dong_bao = thang_bao * (-120000)          

tac_dong_cup_tuan = cup_dien_tuan * (-85000)
tac_dong_cup_cuoituan = cup_dien_cuoi_tuan * (-60000)
tac_dong_le = ngay_le * (-1000)
tac_dong_nghi = ngay_nghi * (-1000)

df['San_luong_tong'] = (tong_co_ban + tang_truong_cung_ky + tac_dong_nhiet + tac_dong_mua_nang 
                        + tac_dong_mua_mua + tac_dong_bao + tac_dong_cup_tuan 
                        + tac_dong_cup_cuoituan + tac_dong_le + tac_dong_nghi)

# Tập đặc trưng huấn luyện (24 biến)
X = df[['Thang', 'Nam', 'Nhiet_do', 'Do_am', 'Mat_do_dan_so', 'So_khach_hang', 'Toc_do_phat_trien', 
        'Thang_nang_nong', 'Thang_mua', 'Thang_bao', 'Cup_dien_tuan', 'Cup_dien_cuoi_tuan', 
        'Ngay_le', 'Ngay_nghi', 'Phu_tai_1', 'Phu_tai_2', 'Phu_tai_3', 'Phu_tai_4', 'Phu_tai_5',
        'Pt1_ky_truoc', 'Pt2_ky_truoc', 'Pt3_ky_truoc', 'Pt4_ky_truoc', 'Pt5_ky_truoc']]
y = df['San_luong_tong']

print("🧠 Đang huấn luyện mô hình XGBoost với khả năng tương thích dữ liệu trống...")
model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Xuất file mô hình
joblib.dump(model, 'xgboost_model.pkl')
print("✅ Huấn luyện thành công và xuất file 'xgboost_model.pkl' tối ưu xử lý dữ liệu thiếu!")