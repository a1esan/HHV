from flask import Flask, request, jsonify
from flask_cors import CORS
import matlab.engine

app = Flask(__name__)
CORS(app)

print("กำลังเชื่อมต่อกับ MATLAB...")
eng = matlab.engine.start_matlab()
print("เชื่อมต่อ MATLAB สำเร็จ!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # 1. รับค่า 6 ตัวแปรจากหน้าเว็บ และคูณน้ำหนักตามสูตรของคุณ
        fw = (float(data.get('fw', 0)) / 100) * 16603.48
        pa = (float(data.get('pa', 0)) / 100) * 15821.31
        pf = (float(data.get('pf', 0)) / 100) * 32763.28
        t  = (float(data.get('t', 0)) / 100)  * 21733.86
        w  = (float(data.get('w', 0)) / 100)  * 16920.80
        lr = (float(data.get('lr', 0)) / 100) * 29259.83
        m  = (float(data.get('m', 0)) / 1)
        
        # . ส่งข้อมูลเข้า MATLAB 
        # 2.ต้องมีครบ 7 ตัว (E_fw, E_pa, E_pf, E_T, E_W, E_Lr และ M)
        eng.workspace['input_data'] = eng.struct({
            'E_fw': fw, 
            'E_pa': pa, 
            'E_pf': pf, 
            'E_T':  t,   # ตัว T ใหญ่
            'E_W':  w,   # ตัว W ใหญ่
            'E_Lr': lr,  # ตัว L ใหญ่ r เล็ก
            'M':    m # ค่าความชื้น 
        })
        eng.eval("T = struct2table(input_data)", nargout=0)
        
        # 3. โหลดไฟล์โมเดล
        eng.eval("S = load('RegressionLearnerSession2.mat')", nargout=0)
        
        # 4.ดึงโมเดลออกมาใช้งาน
        eng.eval("names = fieldnames(S); modelVar = S.(names{1});", nargout=0)
        
        # 5. สั่งคำนวณ
        result = eng.eval("modelVar.predictFcn(T)", nargout=1)
        hhv_value = float(result)

        # ==========================================
        # ส่วนที่เพิ่ม: แสดงค่าใน Terminal
        # ==========================================
        print("\n" + "="*30)
        print(f" ผลการคำนวณ HHV: {hhv_value:.2f} kJ/kg")
        print("="*30 + "\n")

        
        return jsonify({
            'success': True, 
            'hhv': float(result),
            'm_used': m
        })

    except Exception as e:
        print(f"Error logic: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=False)