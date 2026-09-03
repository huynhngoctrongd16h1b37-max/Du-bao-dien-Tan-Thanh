from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

latest_result_df = None

def load_model():
    try:
        return joblib.load('xgboost_model.pkl')
    except:
        return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        thang_val = float(data.get('thang', 7))
        nhiet_do = float(data.get('nhiet_do', 39))
        do_am = float(data.get('do_am', 69))
        toc_do = float(data.get('toc_do', 9.0))
        
        so_ngay_cup_dien = float(data.get('cup_dien_cuoi_tuan') or 0)
        so_ngay_bao = float(data.get('so_ngay_bao') or 0)
        ngay_le = float(data.get('ngay_le') or 0)
        ngay_nghi_t7_cn = float(data.get('ngay_nghi') or 8)
        
        if thang_val in [3, 4, 5, 6]:
            he_so_mua_vu = 1.085
        elif thang_val in [7, 8, 9, 10, 11]:
            he_so_mua_vu = 1.035
        else:
            he_so_mua_vu = 1.050
            
        yeu_to_thoi_tiet = 1.0 + max(0, (nhiet_do - 35) * 0.005) - max(0, (60 - do_am) * 0.002)
        ty_le_tang = (toc_do / 100.0) * he_so_mua_vu * yeu_to_thoi_tiet

        pt1_ky = float(data.get('pt1_ky_truoc') or 1091277)
        pt1_lk = float(data.get('pt1_lien_ke') or 1261088)
        pt1 = float(data.get('pt1') or (pt1_ky * (1 + ty_le_tang) * 0.5 + pt1_lk * 1.01 * 0.5))

        pt2_ky = float(data.get('pt2_ky_truoc') or 1211972)
        pt2_lk = float(data.get('pt2_lien_ke') or 1670640)
        pt2 = float(data.get('pt2') or (pt2_ky * (1 + ty_le_tang) * 0.5 + pt2_lk * 1.01 * 0.5))

        pt3_ky = float(data.get('pt3_ky_truoc') or 238729)
        pt3_lk = float(data.get('pt3_lien_ke') or 345032)
        pt3 = float(data.get('pt3') or (pt3_ky * (1 + ty_le_tang) * 0.5 + pt3_lk * 1.01 * 0.5))

        pt4_ky = float(data.get('pt4_ky_truoc') or 5000143)
        pt4_lk = float(data.get('pt4_lien_ke') or 6415389)
        pt4 = float(data.get('pt4') or (pt4_ky * (1 + ty_le_tang) * 0.5 + pt4_lk * 1.01 * 0.5))

        pt5_ky = float(data.get('pt5_ky_truoc') or 401870)
        pt5_lk = float(data.get('pt5_lien_ke') or 490409)
        pt5 = float(data.get('pt5') or (pt5_ky * (1 + ty_le_tang) * 0.5 + pt5_lk * 1.01 * 0.5))

        thang_nang_nong = 1 if (3 <= thang_val <= 6) else 0
        thang_mua = 1 if (7 <= thang_val <= 11) else 0

        # Đóng gói toàn bộ biến đặc trưng để mô hình AI tự tính toán nội bộ
        df_manual = pd.DataFrame([{
            'Thang': thang_val, 
            'Nam': float(data.get('nam', 2026)),
            'Nhiet_do': nhiet_do, 
            'Do_am': do_am,
            'Mat_do_dan_so': 200, 
            'So_khach_hang': float(data.get('khach_hang', 26625)),
            'Toc_do_phat_trien': toc_do, 
            'Thang_nang_nong': thang_nang_nong,
            'Thang_mua': thang_mua, 
            'Thang_bao': so_ngay_bao,
            'Cup_dien_tuan': 0, 
            'Cup_dien_cuoi_tuan': so_ngay_cup_dien,
            'Ngay_le': ngay_le, 
            'Ngay_nghi': ngay_nghi_t7_cn,
            'Phu_tai_1': pt1, 'Phu_tai_2': pt2, 'Phu_tai_3': pt3, 'Phu_tai_4': pt4, 'Phu_tai_5': pt5,
            'Pt1_ky_truoc': pt1_ky, 'Pt2_ky_truoc': pt2_ky, 'Pt3_ky_truoc': pt3_ky, 'Pt4_ky_truoc': pt4_ky, 'Pt5_ky_truoc': pt5_ky
        }])

        prediction = model.predict(df_manual)[0]
        result = round(float(prediction), 2)
        
        return jsonify({'status': 'success', 'result': f"{result:,.2f}"})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download_template')
def download_template():
    df_template = pd.DataFrame({
        'STT': [1, 2, 3],
        'Loai_Ky': ['Cùng kỳ năm trước', 'Tháng liền kề trước', 'Tháng cần dự báo'],
        'Thang': [7, 5, 7],
        'Nam': [2025, 2026, 2026],
        'Nhiet_do': [39, 38, 39],
        'Do_am': [65, 70, 69],
        'So_khach_hang': [25505, 26570, 26625],
        'Ngay_le': [0, 0, 0],
        'Ngay_nghi': [8, 8, 8],
        'Cup_dien_cuoi_tuan': [1, 2, 2],
        'Thang_bao': [0, 0, 0],
        'Toc_do_phat_trien': [9, 9, 9],
        'Nong_lam_nghiep_thuy_san': [1091277, 1261088, None],
        'Cong_nghiep_Xay_dung': [1211972, 1670640, None],
        'Thuong_nghiep_khach_san_nhahang': [238729, 345032, None],
        'Quan_ly_tieu_dung': [5000143, 6415389, None],
        'Hoat_dong_khac': [401870, 490409, None]
    })
    file_path = 'File_Mau_Du_Bao.xlsx'
    df_template.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)

@app.route('/predict_excel', methods=['POST'])
def predict_excel():
    global latest_result_df
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'status': 'error', 'message': 'Vui lòng chọn file Excel!'})
        
        df = pd.read_excel(file)
        results = []
        
        for i in range(0, len(df), 3):
            group = df.iloc[i:i+3].copy()
            if len(group) < 3:
                for idx, row in group.iterrows():
                    results.append(row)
                break
                
            row_ky = group.iloc[0]
            row_lienke = group.iloc[1]
            row_dubao = group.iloc[2]
            
            thang_val = float(row_dubao.get('Thang', 7))
            nam_val = float(row_dubao.get('Nam', 2026))
            nhiet_do = float(row_dubao.get('Nhiet_do', 39))
            do_am = float(row_dubao.get('Do_am', 69))
            khach_hang = float(row_dubao.get('So_khach_hang', 26625))
            toc_do = float(row_dubao.get('Toc_do_phat_trien', 9))
            
            so_ngay_cup_dien = float(row_dubao.get('Cup_dien_cuoi_tuan', 0))
            so_ngay_bao = float(row_dubao.get('Thang_bao', 0) if 'Thang_bao' in row_dubao else 0)
            ngay_le = float(row_dubao.get('Ngay_le', 0))
            ngay_nghi = float(row_dubao.get('Ngay_nghi', 8))
            
            if thang_val in [3, 4, 5, 6]:
                he_so_mua_vu = 1.085
            elif thang_val in [7, 8, 9, 10, 11]:
                he_so_mua_vu = 1.035
            else:
                he_so_mua_vu = 1.050
                
            yeu_to_thoi_tiet = 1.0 + max(0, (nhiet_do - 35) * 0.005) - max(0, (60 - do_am) * 0.002)
            ty_le = (toc_do / 100.0) * he_so_mua_vu * yeu_to_thoi_tiet
            
            cols = ['Nong_lam_nghiep_thuy_san', 'Cong_nghiep_Xay_dung', 'Thuong_nghiep_khach_san_nhahang', 'Quan_ly_tieu_dung', 'Hoat_dong_khac']
            du_bao_pt = {}
            
            for col in cols:
                val_thuc_te_file = row_dubao.get(col, None)
                if pd.notna(val_thuc_te_file) and str(val_thuc_te_file).strip() != "" and float(val_thuc_te_file) > 0:
                    du_bao_pt[col] = float(val_thuc_te_file)
                else:
                    val_ky = float(row_ky.get(col, 0) or 0)
                    val_lienke = float(row_lienke.get(col, 0) or 0)
                    du_bao_pt[col] = round((val_ky * (1 + ty_le) * 0.5) + (val_lienke * 1.01 * 0.5), 2)
                row_dubao[col] = du_bao_pt[col]

            thang_nang_nong = 1 if (3 <= thang_val <= 6) else 0
            thang_mua = 1 if (7 <= thang_val <= 11) else 0

            df_row = pd.DataFrame([{
                'Thang': thang_val, 'Nam': nam_val,
                'Nhiet_do': nhiet_do, 'Do_am': do_am,
                'Mat_do_dan_so': 200, 'So_khach_hang': khach_hang,
                'Toc_do_phat_trien': toc_do, 'Thang_nang_nong': thang_nang_nong,
                'Thang_mua': thang_mua, 'Thang_bao': so_ngay_bao,
                'Cup_dien_tuan': 0, 'Cup_dien_cuoi_tuan': so_ngay_cup_dien,
                'Ngay_le': ngay_le, 'Ngay_nghi': ngay_nghi,
                'Phu_tai_1': du_bao_pt['Nong_lam_nghiep_thuy_san'], 
                'Phu_tai_2': du_bao_pt['Cong_nghiep_Xay_dung'], 
                'Phu_tai_3': du_bao_pt['Thuong_nghiep_khach_san_nhahang'], 
                'Phu_tai_4': du_bao_pt['Quan_ly_tieu_dung'], 
                'Phu_tai_5': du_bao_pt['Hoat_dong_khac'],
                'Pt1_ky_truoc': float(row_ky.get('Nong_lam_nghiep_thuy_san', 0)), 
                'Pt2_ky_truoc': float(row_ky.get('Cong_nghiep_Xay_dung', 0)), 
                'Pt3_ky_truoc': float(row_ky.get('Thuong_nghiep_khach_san_nhahang', 0)), 
                'Pt4_ky_truoc': float(row_ky.get('Quan_ly_tieu_dung', 0)), 
                'Pt5_ky_truoc': float(row_ky.get('Hoat_dong_khac', 0))
            }])
            
            pred = model.predict(df_row)[0]
            row_ky['Ket_qua_du_bao'] = ""
            row_lienke['Ket_qua_du_bao'] = ""
            row_dubao['Ket_qua_du_bao'] = round(float(pred), 2)
            
            results.append(row_ky)
            results.append(row_lienke)
            results.append(row_dubao)
            
        latest_result_df = pd.DataFrame(results)
        html_table = latest_result_df.to_html(classes='table table-striped table-bordered text-center', index=False)
        return jsonify({'status': 'success', 'html_table': html_table})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download_result_excel')
def download_result_excel():
    global latest_result_df
    if latest_result_df is not None:
        file_path = 'Ket_Qua_Du_Bao.xlsx'
        latest_result_df.to_excel(file_path, index=False)
        return send_file(file_path, as_attachment=True)
    return "Chưa có dữ liệu kết quả dự báo!", 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)